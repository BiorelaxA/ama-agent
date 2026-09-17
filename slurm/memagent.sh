#!/usr/bin/env bash
#SBATCH --job-name=memagent-qwen3-32b-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH -w gpu05
#SBATCH --mem=70G
#SBATCH --output=/home/hongyshen/AMA-Hub/slurm/slurm-%x-%j.out
#SBATCH --error=/home/hongyshen/AMA-Hub/slurm/slurm-%x-%j.err

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd /home/hongyshen/AMA-Hub

# These defaults match the local setup documented in RUNBOOK.md. They can be
# overridden at submission time, for example:
#   sbatch --export=ALL,CONDA_ENV=my_env,EPISODE_IDS=0 slurm/memagent.sh
CONDA_SH="${CONDA_SH:-/home/hongyshen/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-ama_env}"
LLM_CONFIG="${LLM_CONFIG:-configs/qwen3-32B-local.yaml}"
METHOD_CONFIG="${METHOD_CONFIG:-configs/method_configs/memagent.yaml}"
TEST_DIR="${TEST_DIR:-data/test}"
EPISODE_IDS="${EPISODE_IDS:-0,1}"
EVALUATE="${EVALUATE:-True}"

JOB_TAG="${SLURM_JOB_ID:-manual-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/memagent_slurm/$JOB_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-results/memagent_smoke_$JOB_TAG}"
VLLM_LOG="logs/qwen3-32b-vllm.log"
TEST_LOG="$RUN_DIR/test.log"
mkdir -p "$RUN_DIR" "$OUTPUT_DIR"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda initialization script not found: $CONDA_SH" >&2
    exit 1
fi
if [[ ! -f "$LLM_CONFIG" ]]; then
    echo "LLM config not found: $LLM_CONFIG" >&2
    exit 1
fi
if [[ ! -f "$METHOD_CONFIG" ]]; then
    echo "MemAgent config not found: $METHOD_CONFIG" >&2
    exit 1
fi
if [[ ! -f "$TEST_DIR/open_end_qa_set.jsonl" ]]; then
    echo "Test data not found: $TEST_DIR/open_end_qa_set.jsonl" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# Read the model/server values from the same config used by ModelClient so the
# tokenizer path and vLLM endpoint cannot accidentally diverge.
IFS=$'\t' read -r MODEL_PATH VLLM_HOST VLLM_PORT CONFIG_MAX_MODEL_LEN CONFIG_GPU_UTILIZATION < <(
    python -c '
import sys, yaml
c = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
v = c.get("vllm_launch", {})
print(c["model"], c.get("vllm_host", "127.0.0.1"), c.get("vllm_port", 8000),
      v.get("max_model_len", 32000), v.get("gpu_memory_utilization", 0.95), sep="\t")
' "$LLM_CONFIG"
)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$CONFIG_MAX_MODEL_LEN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-$CONFIG_GPU_UTILIZATION}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Qwen3-32B model directory not found: $MODEL_PATH" >&2
    exit 1
fi

# Keep localhost inference traffic away from any configured HTTP proxy.
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

echo "Job ID: $JOB_TAG"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Conda environment: $CONDA_ENV"
echo "Model: $MODEL_PATH"
echo "Episodes: $EPISODE_IDS"
echo "Run logs: $RUN_DIR"
echo "Results: $OUTPUT_DIR"
nvidia-smi

if curl --noproxy localhost,127.0.0.1 -sf \
    "http://$VLLM_HOST:$VLLM_PORT/health" >/dev/null; then
    echo "Port $VLLM_PORT already has a vLLM service; refusing to reuse it." >&2
    exit 1
fi

echo "Starting Qwen3-32B with tensor parallelism across two GPUs..."
if ! CUDA_VISIBLE_DEVICES=0,1 VLLM_LOG="$VLLM_LOG" \
    bash scripts/launch_vllm_32B.sh configs/qwen3-32B-local.yaml; then
    echo "Failed to launch Qwen3-32B. Last vLLM log lines:" >&2
    tail -n 100 "$VLLM_LOG" >&2 || true
    exit 1
fi

# launch_vllm_32B.sh waits for readiness before returning successfully. Verify
# the endpoint once more here so a process that exited immediately afterwards
# cannot be mistaken for a successful deployment.
if ! curl --noproxy localhost,127.0.0.1 -sf \
    "http://$VLLM_HOST:$VLLM_PORT/health" >/dev/null; then
    echo "Qwen3-32B launch command completed, but its health check failed." >&2
    tail -n 100 "$VLLM_LOG" >&2 || true
    exit 1
fi
echo "Qwen3-32B is ready at http://$VLLM_HOST:$VLLM_PORT."

echo "Running the MemAgent test..."
/usr/bin/time -p python src/run.py \
    --llm-server vllm \
    --llm-config configs/qwen3-32B-local.yaml \
    --judge-server vllm \
    --judge-config configs/qwen3-32B-local.yaml \
    --subset openend \
    --method memagent \
    --method-config configs/method_configs/memagent.yaml \
    --test-dir data/test \
    --max-concurrency-episodes 4 \
    --max-concurrency-questions-per-episode 4 \
    --judge-max-concurrency 16 \
    --evaluate "$EVALUATE" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "$TEST_LOG"

echo "MemAgent test completed successfully."
echo "Test log: $TEST_LOG"
echo "Results: $OUTPUT_DIR"
