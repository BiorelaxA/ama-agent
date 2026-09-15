"""MemAgent recurrent-memory baseline adapted to AMA-Bench.

This adapter follows AMA-Bench's fixed-memory evaluation protocol: the full
trajectory is ingested once per episode, and all questions are answered against
the resulting immutable recurrent memory.  The update/final prompts and default
generation settings follow the upstream MemAgent quickstart implementation.
"""

from dataclasses import dataclass
import re
import threading
from typing import Any, List, Optional

from src.method.base_method import BaseMethod


NO_MEMORY = "No previous memory"

MEMORY_UPDATE_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{problem}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

FINAL_ANSWER_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem>
{question}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""


@dataclass(frozen=True)
class MemAgentMemory:
    """The fixed recurrent memory shared by every question in one episode."""

    content: str
    construction_problem: str
    trajectory_tokens: int
    chunks_processed: int


class MemAgentMethod(BaseMethod):
    """Query-independent recurrent memory construction for AMA-Bench."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        client: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ):
        del embedding_engine  # MemAgent does not use an embedding retriever.
        config = self._load_config(config_path)

        self.client = client
        if self.client is None:
            raise ValueError("MemAgent requires the backbone ModelClient")

        self.model_name = config.get("model", "Qwen3-32B")
        self.chunk_size = int(config.get("chunk_size", 5000))
        self.memory_max_tokens = int(config.get("memory_max_tokens", 1024))
        self.final_answer_max_tokens = int(
            config.get("final_answer_max_tokens", 1024)
        )
        self.temperature = float(config.get("temperature", 0.7))
        self.top_p = float(config.get("top_p", 0.95))
        self.enable_thinking = bool(config.get("enable_thinking", True))
        self.strip_thinking_from_memory = bool(
            config.get("strip_thinking_from_memory", True)
        )
        self.trajectory_max_tokens = config.get("trajectory_max_tokens")
        self.construction_problem = config.get(
            "construction_problem", "episode_task"
        )
        self.build_once_per_episode = bool(
            config.get("build_once_per_episode", True)
        )

        if self.chunk_size <= 0:
            raise ValueError("MemAgent chunk_size must be positive")
        if self.memory_max_tokens <= 0 or self.final_answer_max_tokens <= 0:
            raise ValueError("MemAgent generation token limits must be positive")
        if self.construction_problem != "episode_task":
            raise ValueError(
                "This AMA-Bench adapter currently supports only "
                "construction_problem=episode_task"
            )
        if not self.build_once_per_episode:
            raise ValueError(
                "This adapter implements the AMA-Bench fixed-memory protocol; "
                "build_once_per_episode must be true"
            )

        self._tokenizer = None
        self._tokenizer_lock = threading.Lock()

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer

        with self._tokenizer_lock:
            if self._tokenizer is not None:
                return self._tokenizer

            get_local_tokenizer = getattr(self.client, "_get_local_tokenizer", None)
            tokenizer = get_local_tokenizer() if get_local_tokenizer else None
            if tokenizer is None:
                raise RuntimeError(
                    "MemAgent needs a local tokenizer to make exact 5000-token "
                    "chunks. Point llm-config model to the local Qwen3-32B path."
                )
            self._tokenizer = tokenizer
            return tokenizer

    def _encode(self, text: str) -> List[int]:
        return self._get_tokenizer().encode(text, add_special_tokens=False)

    def _decode(self, token_ids: List[int]) -> str:
        return self._get_tokenizer().decode(
            token_ids,
            skip_special_tokens=False,
        )

    def _input_budget(self, output_tokens: int) -> int:
        budget_fn = getattr(self.client, "_prompt_input_budget", None)
        if budget_fn:
            return int(budget_fn(output_tokens))

        llm_config = getattr(self.client, "config", {}) or {}
        launch_config = llm_config.get("vllm_launch", {})
        max_model_len = int(
            llm_config.get("max_model_len")
            or launch_config.get("max_model_len", 32768)
        )
        return max(256, max_model_len - output_tokens - 256)

    def _fit_memory_in_prompt(
        self,
        template: str,
        memory: str,
        output_tokens: int,
        **template_values: str,
    ) -> str:
        """Fit a prompt by discarding the oldest previous-memory tokens first."""
        prompt = template.format(memory=memory, **template_values)
        budget = self._input_budget(output_tokens)
        if len(self._encode(prompt)) <= budget:
            return prompt

        empty_memory_prompt = template.format(memory="", **template_values)
        fixed_tokens = len(self._encode(empty_memory_prompt))
        allowed_memory_tokens = max(0, budget - fixed_tokens)
        memory_ids = self._encode(memory)
        fitted_memory = (
            self._decode(memory_ids[-allowed_memory_tokens:])
            if allowed_memory_tokens
            else ""
        )
        prompt = template.format(memory=fitted_memory, **template_values)

        # Tokenization around template boundaries is not perfectly additive.
        # Remove a few more oldest tokens until the exact encoded prompt fits.
        while fitted_memory and len(self._encode(prompt)) > budget:
            fitted_ids = self._encode(fitted_memory)
            excess = len(self._encode(prompt)) - budget
            fitted_memory = self._decode(fitted_ids[min(len(fitted_ids), excess + 8):])
            prompt = template.format(memory=fitted_memory, **template_values)

        if len(self._encode(prompt)) > budget:
            raise ValueError(
                "MemAgent prompt excluding previous memory exceeds the model "
                "input budget; reduce chunk_size or construction task length"
            )
        return prompt

    @staticmethod
    def _without_thinking(text: str) -> str:
        """Keep only visible content after Qwen-style thinking blocks."""
        text = (text or "").strip()
        if "</think>" in text:
            return text.rsplit("</think>", 1)[-1].strip()
        if "<think>" in text:
            # A generation that stops before </think> has no completed visible
            # memory to carry forward. Returning the prefix (normally empty)
            # makes the caller fail the episode instead of storing hidden
            # reasoning as memory despite strip_thinking_from_memory=true.
            return text.split("<think>", 1)[0].strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _call_model(self, prompt: str, max_tokens: int) -> str:
        return self.client.query(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens,
            enable_thinking=self.enable_thinking,
        )

    def memory_construction(self, traj_text: str, task: str = "") -> MemAgentMemory:
        """Ingest the full trajectory once and construct one recurrent memory."""
        token_ids = self._encode(traj_text)
        original_token_count = len(token_ids)

        if self.trajectory_max_tokens is not None:
            max_trajectory_tokens = int(self.trajectory_max_tokens)
            if max_trajectory_tokens <= 0:
                raise ValueError("trajectory_max_tokens must be null or positive")
            if len(token_ids) > max_trajectory_tokens:
                head = max_trajectory_tokens // 2
                token_ids = token_ids[:head] + token_ids[-(max_trajectory_tokens - head):]

        chunks = [
            token_ids[i:i + self.chunk_size]
            for i in range(0, len(token_ids), self.chunk_size)
        ]
        construction_problem = task.strip() or (
            "Preserve the trajectory's important actions, observations, state "
            "changes, failures, and causal dependencies for future questions."
        )
        memory = NO_MEMORY

        print(
            "MemAgent construction: "
            f"trajectory_tokens={original_token_count}, chunks={len(chunks)}, "
            f"chunk_size={self.chunk_size}"
        )
        for chunk_index, chunk_ids in enumerate(chunks, start=1):
            prompt = self._fit_memory_in_prompt(
                MEMORY_UPDATE_TEMPLATE,
                memory,
                self.memory_max_tokens,
                problem=construction_problem,
                chunk=self._decode(chunk_ids),
            )
            response = self._call_model(prompt, self.memory_max_tokens)
            updated_memory = (
                self._without_thinking(response)
                if self.strip_thinking_from_memory
                else response.strip()
            )
            if not updated_memory:
                raise RuntimeError(
                    f"MemAgent produced empty memory at chunk "
                    f"{chunk_index}/{len(chunks)}"
                )
            memory = updated_memory

        return MemAgentMemory(
            content=memory,
            construction_problem=construction_problem,
            trajectory_tokens=original_token_count,
            chunks_processed=len(chunks),
        )

    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        marker_index = text.rfind("\\boxed")
        if marker_index < 0:
            return None
        open_index = text.find("{", marker_index)
        if open_index < 0:
            return None

        depth = 0
        for index in range(open_index, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[open_index + 1:index].strip()
        return None

    def _extract_answer(self, response: str) -> str:
        visible_response = self._without_thinking(response)
        boxed_answer = self._extract_boxed_answer(visible_response)
        if boxed_answer:
            return boxed_answer

        answer_match = re.search(
            r"Answer\[1\]:\s*(.+)$",
            visible_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            return answer_match.group(1).strip()

        therefore_match = re.search(
            r"Therefore,\s*the answer is\s*(.+)$",
            visible_response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if therefore_match:
            return therefore_match.group(1).strip().rstrip(".")
        return visible_response.strip()

    def memory_retrieve(self, memory: MemAgentMemory, question: str) -> str:
        """Answer one question from the immutable episode-level memory."""
        if not isinstance(memory, MemAgentMemory):
            raise TypeError("MemAgent memory must be a MemAgentMemory instance")

        prompt = self._fit_memory_in_prompt(
            FINAL_ANSWER_TEMPLATE,
            memory.content,
            self.final_answer_max_tokens,
            question=question,
        )
        response = self._call_model(prompt, self.final_answer_max_tokens)
        answer = self._extract_answer(response)
        if not answer:
            raise RuntimeError("MemAgent produced an empty final answer")

        # MemoryQAInterface recognizes this marker and skips its generic second
        # answering call, preserving MemAgent's own final-answer generation.
        return (
            f"<<<AMA_DIRECT_ANSWER>>>{answer}<<<END_AMA_DIRECT_ANSWER>>>\n"
            f"[MemAgent fixed memory; chunks={memory.chunks_processed}, "
            f"trajectory_tokens={memory.trajectory_tokens}]\n"
            f"{memory.content}"
        )
