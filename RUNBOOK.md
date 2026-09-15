# AMA-Bench Baseline 运行命令

本文档记录使用本地 Qwen3-32B 在 AMA-Bench real-world/open-ended 数据集上运行 BM25 和 Qwen3-Embedding-4B baseline 的命令。

## 1. 进入环境

```bash
cd /home/hongyshen/AMA-Hub
conda activate ama_env
mkdir -p logs results
```

当前使用的主要配置：

- 模型：`/home/share/models/Qwen3-32B`
- 模型配置：`configs/qwen3-32B-local.yaml`
- BM25 配置：`configs/method_configs/bm25_config.json`
- 数据集：`data/test/open_end_qa_set.jsonl`
- BM25：`top_k=10`

## 2. 启动本地 vLLM 服务

启动脚本内部使用 `nohup` 将 vLLM 服务放到后台：

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_LOG=logs/qwen3-32b-vllm.log \
  bash scripts/launch_vllm_32B.sh configs/qwen3-32B-local.yaml
```

查看服务日志：

```bash
tail -f logs/qwen3-32b-vllm.log
```

检查健康状态。使用 `--noproxy` 可以避免本机代理把 `localhost` 请求转发到其他端口：

```bash
curl --noproxy localhost,127.0.0.1 -i http://localhost:8056/health
```

检查模型列表：

```bash
curl --noproxy localhost,127.0.0.1 http://localhost:8056/v1/models
```

## 3. 运行10个 episode 进行测试

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method bm25 \
  --method-config configs/method_configs/bm25_config.json \
  --test-dir data/test \
  --episode-ids 0,1,2,3,4,5,6,7,8,9 \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/bm25_10episodes
```

## 4. 首次运行完整 BM25 实验

首次运行不要添加 `--resume`。非续跑模式会清空相同名称的旧 checkpoint，并从头处理全部208个 episode：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method bm25 \
  --method-config configs/method_configs/bm25_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/bm25_full
```

运行过程中，每完成一个 episode 都会立即写入：

```text
results/bm25_full/checkpoint__home_share_models_Qwen3-32B_openend_bm25.jsonl
```

## 5. 中断后续跑

如果完整实验被中断，重新执行相同命令并增加 `--resume`：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method bm25 \
  --method-config configs/method_configs/bm25_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/bm25_full \
  --resume
```

续跑时：

- `status=complete` 的 episode 会被跳过；
- `status=partial` 或 `status=failed` 的 episode 会重新执行；
- 第一阶段全部完成后，会重新进行 LLM-as-a-Judge 评分。

## 6. 将完整评测任务放到后台

首次后台运行：

```bash
nohup bash -lc '
cd /home/hongyshen/AMA-Hub
source /home/hongyshen/miniconda3/etc/profile.d/conda.sh
conda activate ama_env
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method bm25 \
  --method-config configs/method_configs/bm25_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/bm25_full
' > logs/bm25_full.log 2>&1 &

echo $! > logs/bm25_full.pid
```

查看后台评测日志：

```bash
tail -f logs/bm25_full.log
```

检查进程：

```bash
ps -fp "$(cat logs/bm25_full.pid)"
```

如果后台任务中断，使用第5节的命令并在外层采用相同的 `nohup bash -lc '...'` 方式运行即可。

## 7. 仅生成答案，不运行 Judge

如需先完成答案生成，可以增加：

```bash
--evaluate False
```

完整示例：

```bash
python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method bm25 \
  --method-config configs/method_configs/bm25_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --output-dir results/bm25_full \
  --evaluate False
```

---

# Qwen3-Embedding-4B（16K + query instruction）

## 8. 本次 Embedding 配置

`configs/method_configs/embedding_config.json` 当前配置为：

- 检索模型：`qwen3-embedding-4B`
- `top_k=5`
- 检索：FAISS cosine similarity
- query instruction：`Given a web search query, retrieve relevant passages that answer the query`
- 问题的实际编码格式：`Instruct: <instruction>\nQuery:<question>`
- 轨迹文档块不加 instruction
- 每个轨迹块最多使用 `16384 tokens` 进行 embedding；截断由 vLLM 使用模型 tokenizer 精确执行，不再按字符数估算。

注意：这是一组新的实验设置，不要对之前使用 512-token/无 instruction 生成的 checkpoint 直接执行 `--resume`。

## 9. 启动 Qwen3-Embedding-4B 服务

先保持第2节的 Qwen3-32B 服务运行。下面示例把 embedding 模型放在 GPU 2；如 GPU 编号不同，修改 `EMBEDDING_GPU`。

```bash
cd /home/hongyshen/AMA-Hub
conda activate ama_env
mkdir -p logs results

EMBEDDING_GPU=2
CUDA_VISIBLE_DEVICES="$EMBEDDING_GPU" \
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/hongyshen/AMA-Hub/models/Qwen3-Embedding-4B \
  --served-model-name qwen3-embedding-4B \
  --runner pooling \
  --host localhost \
  --port 8003 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  > logs/qwen3-embedding-4b-vllm-16k.log 2>&1 &

echo $! > logs/qwen3-embedding-4b-vllm-16k.pid
```

这里保留了之前为 vLLM 0.10.1.1 pooling engine 使用的 `--no-enable-prefix-caching` 和 `--no-enable-chunked-prefill` 稳定性设置。

查看启动日志：

```bash
tail -f logs/qwen3-embedding-4b-vllm-16k.log
```

检查服务：

```bash
curl --noproxy localhost,127.0.0.1 -i http://localhost:8003/health
curl --noproxy localhost,127.0.0.1 http://localhost:8003/v1/models
```

发送一次 embedding 请求进行冒烟测试：

```bash
curl --noproxy localhost,127.0.0.1 http://localhost:8003/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embedding-4B",
    "input": ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What happened in the trajectory?"],
    "truncate_prompt_tokens": 16384
  }'
```

## 10. 先运行 10 个 episode

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method embedding \
  --method-config configs/method_configs/embedding_config.json \
  --test-dir data/test \
  --episode-ids 0,1,2,3,4,5,6,7,8,9 \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/embedding_16k_instruction_10episodes
```

## 11. 首次运行完整 Embedding 实验

首次运行不要添加 `--resume`：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method embedding \
  --method-config configs/method_configs/embedding_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/embedding_16k_instruction_full
```

## 12. Embedding 实验中断后续跑

只有在第11节这一新输出目录已经产生 checkpoint 后，才使用：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method embedding \
  --method-config configs/method_configs/embedding_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/embedding_16k_instruction_full \
  --resume
```

如果 embedding 服务因显存不足退出，先将 `configs/method_configs/embedding_config.json` 中的 `batch_size` 从 8 降到 2 或 1；这只会降低 embedding 吞吐量，不会改变 `top_k` 或 16K 截断规则。

## 13. 将完整 Embedding 实验放到后台

在 Qwen3-32B 和 Qwen3-Embedding-4B 两个服务都通过 health check 后执行：

```bash
nohup bash -lc '
cd /home/hongyshen/AMA-Hub
source /home/hongyshen/miniconda3/etc/profile.d/conda.sh
conda activate ama_env
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method embedding \
  --method-config configs/method_configs/embedding_config.json \
  --test-dir data/test \
  --max-concurrency-episodes 2 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/embedding_16k_instruction_full
' > logs/embedding_16k_instruction_full.log 2>&1 &

echo $! > logs/embedding_16k_instruction_full.pid
```

查看运行状态：

```bash
tail -f logs/embedding_16k_instruction_full.log
ps -fp "$(cat logs/embedding_16k_instruction_full.pid)"
```

后台任务中断后，重新执行上面命令时，将命令最后一行替换为：

```bash
  --output-dir results/embedding_16k_instruction_full \
  --resume
```

---

# AMA-Agent（causal=True）

## 14. AMA-Agent 当前复现配置

本节按照当前仓库中 `causal=True` 的 AMA-Agent 实现运行 real-world/open-ended 全量测试。使用：

- Memory 构建、检索判断和回答模型：`/home/share/models/Qwen3-32B`
- LLM-as-a-Judge：`/home/share/models/Qwen3-32B`
- Embedding 模型：`qwen3-embedding-4B`
- `top_k=5`
- `causal=true`
- 方法配置：`configs/ama_agent_causal.yaml`
- 数据集：`data/test/open_end_qa_set.jsonl`（208 episodes、2496 QA pairs）

注意：该命令复现的是当前公开代码中 `causal=True` 的实际行为。当前检索代码没有使用 `memory["causal_graph"]` 做真正的因果边遍历；embedding 失败时还可能静默回退到简化词法检索。因此运行前必须确认 embedding 服务正常。

确认方法配置：

```bash
sed -n '1,200p' configs/ama_agent_causal.yaml
```

输出中必须至少包含：

```yaml
top_k: 5
causal: true
```

## 15. 启动 AMA-Agent 所需的两个模型服务

进入项目和环境：

```bash
cd /home/hongyshen/AMA-Hub
conda activate ama_env
mkdir -p logs results
```

在 GPU 0、1 上启动 Qwen3-32B。脚本会使用 `nohup` 放到后台：

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_LOG=logs/qwen3-32b-vllm.log \
  bash scripts/launch_vllm_32B.sh configs/qwen3-32B-local.yaml
```

在 GPU 2 上启动 Qwen3-Embedding-4B：

```bash
CUDA_VISIBLE_DEVICES=2 \
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/hongyshen/AMA-Hub/models/Qwen3-Embedding-4B \
  --served-model-name qwen3-embedding-4B \
  --runner pooling \
  --host localhost \
  --port 8003 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.8 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  > logs/qwen3-embedding-4b-ama-agent.log 2>&1 &

echo $! > logs/qwen3-embedding-4b-ama-agent.pid
```

查看两个服务的日志：

```bash
tail -f logs/qwen3-32b-vllm.log
```

```bash
tail -f logs/qwen3-embedding-4b-ama-agent.log
```

确认两个服务均已启动：

```bash
curl --noproxy localhost,127.0.0.1 -i http://localhost:8056/health
curl --noproxy localhost,127.0.0.1 http://localhost:8056/v1/models

curl --noproxy localhost,127.0.0.1 -i http://localhost:8003/health
curl --noproxy localhost,127.0.0.1 http://localhost:8003/v1/models
```

Embedding 服务返回的模型 ID 必须与 `configs/ama_agent_causal.yaml` 中的 `model_name: qwen3-embedding-4B` 完全一致。发送一次真实 embedding 请求：

```bash
curl --noproxy localhost,127.0.0.1 http://localhost:8003/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-embedding-4B",
    "input": ["What happened in the trajectory?"],
    "truncate_prompt_tokens": 16384
  }'
```

## 16. AMA-Agent smoke test

先运行两个 episode。使用独立输出目录，不能从其他 AMA-Agent 配置产生的 checkpoint 续跑：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method ama_agent \
  --method-config configs/ama_agent_causal.yaml \
  --test-dir data/test \
  --episode-ids 0,1 \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 2 \
  --judge-max-concurrency 4 \
  --output-dir results/ama_agent_causal_smoke
```

确认日志中没有 `Status=failed`、`Status=partial`、连接错误或空答案后，再执行全量实验。

## 17. 首次运行完整 AMA-Agent 实验

首次运行不要添加 `--resume`：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method ama_agent \
  --method-config configs/ama_agent_causal.yaml \
  --test-dir data/test \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/ama_agent_causal_full
```

如果显存和服务稳定，可以把 `--max-concurrency-episodes` 从 `1` 提高到 `2`。并发数只影响吞吐和显存压力，不改变 `top_k` 或 `causal` 配置。

## 18. AMA-Agent 中断后续跑

只能对第17节同一配置和同一输出目录产生的 checkpoint 使用 `--resume`：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method ama_agent \
  --method-config configs/ama_agent_causal.yaml \
  --test-dir data/test \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/ama_agent_causal_full \
  --resume
```

续跑会跳过 `status=complete` 且答案数量完整的 episode，并重新运行 `partial` 或 `failed` 的 episode。

## 19. 将完整 AMA-Agent 实验放到后台

两个模型服务通过 health check 后，首次后台运行使用：

```bash
nohup bash -lc '
cd /home/hongyshen/AMA-Hub
source /home/hongyshen/miniconda3/etc/profile.d/conda.sh
conda activate ama_env
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method ama_agent \
  --method-config configs/ama_agent_causal.yaml \
  --test-dir data/test \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/ama_agent_causal_full
' > logs/ama_agent_causal_full.log 2>&1 &

echo $! > logs/ama_agent_causal_full.pid
```

查看后台运行状态：

```bash
tail -f logs/ama_agent_causal_full.log
ps -fp "$(cat logs/ama_agent_causal_full.pid)"
```

后台任务中断后，在 Python 命令末尾增加 `--resume`，其他参数和输出目录保持不变。

---

# MemAgent（固定 episode-level memory）

## 20. MemAgent 接入配置

AMA-Hub 中注册的方法名为 `memagent`，配置文件为：

```text
configs/method_configs/memagent.yaml
```

当前配置：

```yaml
model: "Qwen3-32B"
chunk_size: 5000
memory_max_tokens: 1024
final_answer_max_tokens: 1024
temperature: 0.7
top_p: 0.95
enable_thinking: true
strip_thinking_from_memory: true
trajectory_max_tokens: null
construction_problem: "episode_task"
build_once_per_episode: true
```

实现遵循 AMA-Bench 的固定-memory协议：每个 episode 的完整轨迹只构建一次 memory，随后12个QA共享该 memory。构建时使用 episode 的 `task` 填充上游 MemAgent prompt 的 `<problem>`；轨迹使用 Qwen3-32B tokenizer 精确切成5000-token块。每轮只将去除 thinking 后的可见 memory 传给下一轮。

该适配器复用了 `methods/memory_agent/MemAgent` 中的 recurrent update 和 final-answer prompt，不需要安装上游训练框架的 Ray、verl，也不需要启动 Qwen3-Embedding-4B。

## 21. 启动并检查 Qwen3-32B

```bash
cd /home/hongyshen/AMA-Hub
conda activate ama_env
mkdir -p logs results

CUDA_VISIBLE_DEVICES=0,1 VLLM_LOG=logs/qwen3-32b-vllm.log \
  bash scripts/launch_vllm_32B.sh configs/qwen3-32B-local.yaml
```

检查服务：

```bash
curl --noproxy localhost,127.0.0.1 -i http://localhost:8056/health
curl --noproxy localhost,127.0.0.1 http://localhost:8056/v1/models
```

MemAgent 的 memory 更新和最终回答请求会显式向 vLLM 传入：

```text
temperature=0.7
top_p=0.95
max_tokens=1024
chat_template_kwargs.enable_thinking=true
```

Judge 仍由同一 Qwen3-32B 服务执行。

## 22. MemAgent smoke test

先运行两个 episode：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method memagent \
  --method-config configs/method_configs/memagent.yaml \
  --test-dir data/test \
  --episode-ids 0,1 \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 2 \
  --judge-max-concurrency 4 \
  --output-dir results/memagent_smoke
```

日志中每个 episode 应只出现一次类似下面的 memory 构建记录：

```text
MemAgent construction: trajectory_tokens=..., chunks=..., chunk_size=5000
```

确认没有 `Status=failed`、`Status=partial` 或 `MemAgent produced empty memory` 后再执行全量测试。

## 23. 首次运行完整 MemAgent 实验

首次运行不要添加 `--resume`：

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method memagent \
  --method-config configs/method_configs/memagent.yaml \
  --test-dir data/test \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/memagent_full
```

服务稳定后可以将 `--max-concurrency-episodes` 提高到 `2`。由于一个 episode 内部的 recurrent memory 更新有前后依赖，同一 episode 的5000-token chunks始终按顺序处理；只有不同 episode 和最终 QA 可以并发。

## 24. MemAgent 中断后续跑

```bash
/usr/bin/time -p python src/run.py \
  --llm-server vllm \
  --llm-config configs/qwen3-32B-local.yaml \
  --judge-server vllm \
  --judge-config configs/qwen3-32B-local.yaml \
  --subset openend \
  --method memagent \
  --method-config configs/method_configs/memagent.yaml \
  --test-dir data/test \
  --max-concurrency-episodes 1 \
  --max-concurrency-questions-per-episode 4 \
  --judge-max-concurrency 8 \
  --output-dir results/memagent_full \
  --resume
```

不要使用其他方法或其他 MemAgent 配置生成的 checkpoint 续跑。配置发生变化时应更换新的 `--output-dir`。
