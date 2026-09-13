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
VLLM_LOG=logs/qwen3-32b-vllm.log \
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
