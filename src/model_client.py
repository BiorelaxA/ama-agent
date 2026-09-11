import json
import argparse
import sys
import re
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import yaml

class ModelClient:
    """Unified client for different model providers."""

    def __init__(self, config_path: str, server_type: str = "api", **kwargs):
        """
        Initialize model client.

        Args:
            config_path: Path to YAML config file containing provider, model, api_key, etc.
            server_type: Server type ("api" or "vllm")
            **kwargs: Additional provider-specific arguments (e.g., host, port for vllm)
        """
        # Load config file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

        # Extract provider and model from config
        self.provider = config.get('provider', '').lower()
        self.model = config.get('model', '')
        self.config = config
        vllm_launch = config.get('vllm_launch', {})
        self._max_model_len = (
            config.get('max_model_len')
            or vllm_launch.get('max_model_len', 131072)
        )
        self._context_safety_margin = int(config.get('context_safety_margin', 256))
        self._max_context_truncations = int(config.get('max_context_truncations', 6))
        self._local_tokenizer = None
        self._local_tokenizer_initialized = False
        self._local_tokenizer_lock = threading.Lock()

        # For vllm server type, override provider
        if server_type == "vllm":
            self.provider = "custom"
            # Use host/port from kwargs or config (support both 'host' and 'vllm_host' naming)
            host = config.get('vllm_host') or config.get('host', 'localhost')
            port = config.get('vllm_port') or config.get('port', 8000)
            config["base_url"] = f"http://{host}:{port}/v1"

        self._timeout = float(config.get('timeout', 600.0))
        self.client = self._initialize_client()

    def _get_local_tokenizer(self):
        """Load a tokenizer only when ``model`` points to a local checkpoint.

        Remote/API model names must not trigger an implicit model download.  A
        local tokenizer lets us enforce the real token budget before sending a
        request to vLLM instead of relying on a chars-per-token estimate.
        """
        if self._local_tokenizer_initialized:
            return self._local_tokenizer

        with self._local_tokenizer_lock:
            if self._local_tokenizer_initialized:
                return self._local_tokenizer
            if not self.model or not Path(self.model).exists():
                self._local_tokenizer_initialized = True
                return None
            try:
                from transformers import AutoTokenizer

                self._local_tokenizer = AutoTokenizer.from_pretrained(
                    self.model,
                    local_files_only=True,
                    trust_remote_code=bool(self.config.get('trust_remote_code', False)),
                )
            except Exception as exc:
                print(f"Warning: could not load local tokenizer for context budgeting: {exc}")
                self._local_tokenizer = None
            self._local_tokenizer_initialized = True
            return self._local_tokenizer

    def _prompt_input_budget(self, max_tokens: int) -> int:
        """Return a conservative input budget for a chat request."""
        return max(
            256,
            int(self._max_model_len) - int(max_tokens) - self._context_safety_margin,
        )

    def _truncate_prompt_to_budget(
        self,
        prompt: str,
        max_tokens: int,
        error_str: str = "",
    ) -> tuple[str, bool, Optional[int]]:
        """Keep the beginning and end of a prompt within the model budget.

        The question and answer instructions live at the end of AMA-Bench
        prompts, so head/tail truncation retains them while also preserving the
        highest-ranked retrieved chunks at the beginning.  Returns the new
        prompt, whether it changed, and the measured/estimated original token
        count.
        """
        marker = "\n...[truncated to fit model context]...\n"
        target_tokens = self._prompt_input_budget(max_tokens)
        tokenizer = self._get_local_tokenizer()

        # vLLM currently reports "request has N input tokens" while some APIs
        # use "prompt contains at least N input tokens".  The reported count
        # also covers small differences introduced by a server-side chat
        # template, which the locally encoded raw prompt cannot see.
        count_match = re.search(
            r'(?:request has|prompt contains(?: at least)?)\s+([\d,]+)\s+input tokens',
            error_str,
            flags=re.IGNORECASE,
        )
        reported_tokens = (
            int(count_match.group(1).replace(',', ''))
            if count_match
            else None
        )

        if tokenizer is not None:
            token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            original_tokens = len(token_ids)
            if error_str:
                if reported_tokens and reported_tokens > target_tokens:
                    # Translate the server-observed excess back into the local
                    # tokenizer's scale and leave another 10% safety margin.
                    target_tokens = min(
                        target_tokens,
                        max(
                            256,
                            int(
                                original_tokens
                                * target_tokens
                                / reported_tokens
                                * 0.9
                            ),
                        ),
                    )
                elif original_tokens <= target_tokens:
                    # The API rejected a prompt that looked valid locally.
                    # Force monotonic progress instead of retrying unchanged.
                    target_tokens = max(256, int(original_tokens * 0.8))

            if original_tokens <= target_tokens:
                return prompt, False, original_tokens

            marker_ids = tokenizer.encode(marker, add_special_tokens=False)
            content_budget = max(2, target_tokens - len(marker_ids))
            head_tokens = content_budget // 2
            tail_tokens = content_budget - head_tokens
            truncated = (
                tokenizer.decode(token_ids[:head_tokens], skip_special_tokens=False)
                + marker
                + tokenizer.decode(token_ids[-tail_tokens:], skip_special_tokens=False)
            )
            return truncated, truncated != prompt, original_tokens

        input_tokens = (
            reported_tokens
            if reported_tokens is not None
            else max(1, len(prompt) // 4)
        )
        if input_tokens <= target_tokens and not error_str:
            return prompt, False, input_tokens

        # Leave an extra 10% margin because character density can vary sharply
        # between the head/middle/tail of HTML, JSON, and source-code prompts.
        scale = min(0.9, (target_tokens / max(input_tokens, 1)) * 0.9)
        marker_len = len(marker)
        content_chars = max(2, int(len(prompt) * scale) - marker_len)
        content_chars = min(content_chars, max(2, len(prompt) - marker_len - 1))
        head_chars = content_chars // 2
        tail_chars = content_chars - head_chars
        truncated = prompt[:head_chars] + marker + prompt[-tail_chars:]
        return truncated, len(truncated) < len(prompt), input_tokens

    def _initialize_client(self):
        """Initialize provider-specific client."""
        if self.provider == "custom":
            from openai import OpenAI
            base_url = self.config.get("base_url")
            api_key = self.config.get("api_key", "EMPTY")
            return OpenAI(base_url=base_url, api_key=api_key, timeout=self._timeout)

        elif self.provider == "openai":
            from openai import OpenAI
            # Use api_key from config if provided, otherwise will use OPENAI_API_KEY env var
            api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAI(api_key=api_key, timeout=self._timeout)
            else:
                # Let OpenAI SDK handle the API key (will use OPENAI_API_KEY env var)
                return OpenAI(timeout=self._timeout)

        elif self.provider == "deepseek":
            from openai import OpenAI
            api_key = self.config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API key not found in config or DEEPSEEK_API_KEY environment variable")
            return OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=self._timeout)

        elif self.provider == "gemini":
            import google.generativeai as genai
            api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found in config or GOOGLE_API_KEY environment variable")
            genai.configure(api_key=api_key)
            return genai

        elif self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic
            api_key = self.config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return Anthropic(api_key=api_key)
            else:
                # Let Anthropic SDK handle the API key (will use ANTHROPIC_API_KEY env var)
                return Anthropic()

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def query(self, prompt: str, temperature: float = 0.0, max_tokens: int = 4096, max_retries: int = 3, system: Optional[str] = None) -> str:
        """Query model with prompt with retry logic for rate limits."""
        context_truncations = 0
        attempt = 0

        # For a local checkpoint, enforce the exact token budget proactively.
        original_prompt_len = len(prompt)
        prompt, was_truncated, measured_tokens = self._truncate_prompt_to_budget(
            prompt,
            max_tokens,
        )
        if was_truncated:
            context_truncations += 1
            print(
                "Context budget exceeded; pre-truncating prompt "
                f"({measured_tokens} tokens, {original_prompt_len} chars) -> "
                f"{len(prompt)} chars before request..."
            )

        while attempt < max_retries:
            try:
                if self.provider in ["custom", "deepseek"]:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "openai":
                    # gpt-5 family: hidden chain-of-thought tokens count against
                    # max_completion_tokens, so the chat.completions path can
                    # emit empty visible output even with a generous budget.
                    # Use the Responses API with reasoning_effort=minimal, the
                    # same convention as simpleqa / aa-lcr / Harbor parity.
                    if self.model.startswith("gpt-5"):
                        reasoning_effort = self.config.get("reasoning_effort", "minimal")
                        response = self.client.responses.create(
                            model=self.model,
                            input=prompt,
                            max_output_tokens=max_tokens,
                            reasoning={"effort": reasoning_effort},
                        )
                        return (response.output_text or "").strip()
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=max_tokens,
                        )
                        return response.choices[0].message.content.strip()
                    except Exception as e:
                        if "max_completion_tokens" in str(e) or "unsupported_parameter" in str(e):
                            response = self.client.chat.completions.create(
                                model=self.model,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                            return response.choices[0].message.content.strip()
                        else:
                            raise

                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model)
                    response = model.generate_content(
                        prompt,
                        generation_config=self.client.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens,
                        )
                    )
                    return response.text.strip()

                elif self.provider in ["anthropic", "claude"]:
                    request_params = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if system:
                        request_params["system"] = system
                    response = self.client.messages.create(**request_params)
                    if not response.content:
                        if hasattr(response, 'stop_reason') and response.stop_reason == 'refusal':
                            raise ValueError(f"Claude refused to respond. This may be due to content policy. Stop reason: {response.stop_reason}")
                        raise ValueError(f"Empty response content from Claude API. Response: {response}")
                    if not hasattr(response.content[0], 'text'):
                        raise ValueError(f"Response content has no text attribute. Content type: {type(response.content[0])}, Content: {response.content[0]}")
                    return response.content[0].text.strip()

                else:
                    raise ValueError(f"Query not implemented for provider: {self.provider}")

            except Exception as e:
                error_str = str(e)
                # Don't retry on refusals or permanent errors
                if "refused" in error_str.lower() or "refusal" in error_str.lower():
                    raise
                # Context errors shrink the prompt without consuming a normal
                # transient-error retry attempt.
                is_context_length_error = (
                    ("400" in error_str or "BadRequestError" in error_str)
                    and ("context length" in error_str.lower() or "context_length" in error_str.lower()
                         or "maximum context" in error_str.lower() or "input_tokens" in error_str.lower())
                )
                if is_context_length_error:
                    if context_truncations >= self._max_context_truncations:
                        raise
                    old_len = len(prompt)
                    prompt, changed, measured_tokens = self._truncate_prompt_to_budget(
                        prompt,
                        max_tokens,
                        error_str=error_str,
                    )
                    if not changed:
                        raise
                    context_truncations += 1
                    print(
                        "Context length exceeded; truncating prompt "
                        f"({measured_tokens} tokens, {old_len} chars) -> "
                        f"{len(prompt)} chars, retry "
                        f"{context_truncations}/{self._max_context_truncations}..."
                    )
                    continue
                # Don't retry on other 400/client errors (permanent)
                if "400" in error_str or "BadRequestError" in error_str:
                    raise
                # Check if it's a rate limit error that should be retried
                attempt += 1
                if attempt >= max_retries:
                    raise
                if "rate" in error_str.lower() or "429" in error_str:
                    wait_time = 2 ** (attempt - 1)
                    print(f"Rate limit hit, retrying in {wait_time}s... (attempt {attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    wait_time = min(2 ** attempt, 30)
                    print(f"Error: {error_str}")
                    print(f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)

        raise RuntimeError(f"Failed after {max_retries} retries")
