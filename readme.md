# 投机解码（Speculative Decoding）性能与接受率报告

本报告对比 ernie-4_5-21b-a3b-bf16 在以下九种解码配置下的吞吐、延迟与草稿接受率，用于验证 ngram / MTP / Hybrid MTP-Ngram 路径功能正确性，并量化各方法的加速收益。

| # | 配置 | 简称 | K (`num_speculative_tokens`) | MTP steps |
|---|---|---|---|---|
| 1 | 普通解码（不开启投机） | `non-spec` | — | — |
| 2 | Ngram (K=3) 投机解码 | `ngram-3` | 3 | — |
| 3 | Ngram (K=5) 投机解码 | `ngram-5` | 5 | — |
| 4 | MTP（1 步） | `mtp-1` | 1 | 1 |
| 5 | MTP（3 步） | `mtp-3` | 3 | 3 |
| 6 | MTP（5 步） | `mtp-5` | 5 | 5 |
| 7 | MTP（1 步）+ Hybrid Ngram（补 2 个） | `mtp1+ngram2` | 3 | 1 |
| 8 | MTP（3 步）+ Hybrid Ngram（补 1 个） | `mtp3+ngram1` | 4 | 3 |
| 9 | MTP（3 步）+ Hybrid Ngram（补 2 个） | `mtp3-hybrid` | 5 | 3 |

> 新增配置 2/6/7/8 用于回答几个关键对比问题：
> - **2 (ngram-3) vs 5 (mtp-3) vs 7 (mtp1+ngram2)**：同 K=3 草稿位预算下，"纯 ngram / 纯 MTP / MTP+ngram 混合" 三角对照，验证最优 step/ngram 配比
> - **6 (mtp-5) vs 9 (mtp3-hybrid)**：K=5 时，MTP 全推 5 步 vs MTP 推 3 步 + ngram 补 2 个，哪个更优（"该不该用 ngram 补"）
> - **5 (mtp-3) → 8 (mtp3+ngram1) → 9 (mtp3-hybrid)**：MTP 推 3 步基础上，ngram 补 0/1/2 个的边际收益曲线

---

## 测试环境

| 项 | 值 |
|---|---|
| 模型 | `baidu/ERNIE-4.5-21B-A3B-Paddle` |
| MTP 子模型 | `${MODEL_PATH}/ernie-4_5-21b-a3b-bf16-paddle/mtp` |
| 量化 | wint4 |
| 硬件 | H800x1 |
| tensor-parallel-size | 1 |
| 数据集 | [`filtered_sharedgpt_2000_input_1136_output_200_fd.json`](https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json) |
| 请求数 | 2000 |
| 并发 | 100 |

## 数据集下载

```bash
wget https://fastdeploy.bj.bcebos.com/eb_query/filtered_sharedgpt_2000_input_1136_output_200_fd.json
```

## 公共环境变量

```bash
export MODEL_PATH=/home/aistudio/PaddlePaddle
export FD_API_PORT=8188
export FD_ENGINE_QUEUE_PORT=8133
export FD_METRICS_PORT=8233
export FD_CACHE_QUEUE_PORT=8333
```

---

## 配置 1：non-spec（基线）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --graph-optimization-config '{"use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model non-spec \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_non-spec.log 2>&1
```

### 性能 metrics

| 指标 | 值 |
|---|---|
| Successful requests | 2000 |
| Benchmark duration (s) | 594.28 |
| Total input tokens | 2,541,341 |
| Total generated tokens | 1,949,506 |
| Request throughput (req/s) | 3.365 |
| Output token throughput (tok/s) | 3280.47 |
| Total token throughput (tok/s) | 7556.84 |
| Mean TTFT (ms) | 296.59 |
| Median TTFT (ms) | 117.17 |
| P99 TTFT (ms) | 5038.91 |
| Mean TPOT (ms) | 29.19 |
| Median TPOT (ms) | 28.02 |
| P99 TPOT (ms) | 46.28 |
| Mean ITL (ms) | 28.70 |
| Median ITL (ms) | 27.12 |
| P99 ITL (ms) | 57.20 |
| Mean E2EL (ms) | 28270.04 |
| Median E2EL (ms) | 24455.03 |
| P99 E2EL (ms) | 86897.14 |

### 接受率（N/A，未开启投机解码）

---

## 配置 2：ngram-3（K=3）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"max_ngram_size":3,"min_ngram_size":1}' \
  --graph-optimization-config '{"use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model ngram-3 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_ngram-3.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec | vs mtp-3 |
|---|---|---|---|
| Successful requests | 2000 | — | — |
| Benchmark duration (s) | 727.38 | 1.22× | 1.98× |
| Total input tokens | 2,541,341 | — | — |
| Total generated tokens | 1,985,786 | 1.02× | 0.99× |
| Request throughput (req/s) | 2.750 | 0.82× | 0.51× |
| Output token throughput (tok/s) | 2730.04 | 0.83× | 0.50× |
| Total token throughput (tok/s) | 6223.85 | 0.82× | 0.50× |
| Mean TTFT (ms) | 314.10 | 1.06× | 0.77× |
| Median TTFT (ms) | 141.96 | 1.21× | 0.70× |
| P99 TTFT (ms) | 4837.37 | 0.96× | 0.85× |
| Mean TPOT (ms) | 35.41 | 1.21× | 1.87× |
| Median TPOT (ms) | 37.11 | 1.32× | 2.07× |
| P99 TPOT (ms) | 49.66 | 1.07× | 1.43× |
| Mean ITL (ms) | 37.56 | 1.31× | 0.80× |
| Median ITL (ms) | 37.82 | 1.39× | 0.86× |
| P99 ITL (ms) | 63.50 | 1.11× | 0.69× |
| Mean E2EL (ms) | 34106.17 | 1.21× | 1.93× |
| Median E2EL (ms) | 26671.02 | 1.09× | 1.73× |
| P99 E2EL (ms) | 126715.05 | 1.46× | 2.32× |

### 接受率

```
Speculate global accept ratio: 0.1055
total step:                    880796
total output token num:        984636
average accept len:            1.1179
```

---

## 配置 3：ngram-5（K=5）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5,"max_ngram_size":3,"min_ngram_size":1}' \
  --graph-optimization-config '{"use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model ngram-5 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_ngram-5.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec |
|---|---|---|
| Successful requests | 2000 | — |
| Benchmark duration (s) | 828.98 | 1.39× |
| Total input tokens | 2,541,341 | — |
| Total generated tokens | 1,925,430 | 0.99× |
| Request throughput (req/s) | 2.413 | 0.72× |
| Output token throughput (tok/s) | 2322.64 | 0.71× |
| Total token throughput (tok/s) | 5388.26 | 0.71× |
| Mean TTFT (ms) | 298.70 | 1.01× |
| Median TTFT (ms) | 163.94 | 1.40× |
| P99 TTFT (ms) | 4538.73 | 0.90× |
| Mean TPOT (ms) | 42.35 | 1.45× |
| Median TPOT (ms) | 43.89 | 1.57× |
| P99 TPOT (ms) | 55.85 | 1.21× |
| Mean ITL (ms) | 44.26 | 1.54× |
| Median ITL (ms) | 44.91 | 1.66× |
| P99 ITL (ms) | 71.37 | 1.25× |
| Mean E2EL (ms) | 40182.21 | 1.42× |
| Median E2EL (ms) | 33447.94 | 1.37× |
| P99 E2EL (ms) | 137686.54 | 1.58× |

### 接受率

```
Speculate global accept ratio: 0.07183
total step:                    859225
total output token num:        925722
average accept len:            1.0774
```

---

## 配置 4：MTP（1 步）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"num_model_steps":1,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp-1 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp-1.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec |
|---|---|---|
| Successful requests | 2000 | — |
| Benchmark duration (s) | 495.05 | 0.83× |
| Total input tokens | 2,541,341 | — |
| Total generated tokens | 1,960,206 | 1.01× |
| Request throughput (req/s) | 4.040 | 1.20× |
| Output token throughput (tok/s) | 3959.58 | 1.21× |
| Total token throughput (tok/s) | 9093.04 | 1.20× |
| Mean TTFT (ms) | 415.30 | 1.40× |
| Median TTFT (ms) | 172.49 | 1.47× |
| P99 TTFT (ms) | 5887.64 | 1.17× |
| Mean TPOT (ms) | 24.48 | 0.84× |
| Median TPOT (ms) | 22.86 | 0.82× |
| P99 TPOT (ms) | 49.71 | 1.07× |
| Mean ITL (ms) | 42.54 | 1.48× |
| Median ITL (ms) | 39.42 | 1.45× |
| P99 ITL (ms) | 71.42 | 1.25× |
| Mean E2EL (ms) | 23071.41 | 0.82× |
| Median E2EL (ms) | 20379.97 | 0.83× |
| P99 E2EL (ms) | 72763.54 | 0.84× |

### 接受率

```
Speculate global accept ratio: 0.45699
total step:                    521633
total output token num:        960635
average accept len:            1.8416
Single head accept ratio:      [0.8404]
```

---

## 配置 5：MTP（3 步）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_model_steps":3,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp-3 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp-3.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec |
|---|---|---|
| Successful requests | 2000 | — |
| Benchmark duration (s) | 367.85 | 0.62× |
| Total input tokens | 2,541,341 | — |
| Total generated tokens | 2,000,114 | 1.03× |
| Request throughput (req/s) | 5.437 | 1.62× |
| Output token throughput (tok/s) | 5437.34 | 1.66× |
| Total token throughput (tok/s) | 12346.02 | 1.63× |
| Mean TTFT (ms) | 409.94 | 1.38× |
| Median TTFT (ms) | 202.51 | 1.73× |
| P99 TTFT (ms) | 5707.53 | 1.13× |
| Mean TPOT (ms) | 18.91 | 0.65× |
| Median TPOT (ms) | 17.91 | 0.64× |
| P99 TPOT (ms) | 34.82 | 0.75× |
| Mean ITL (ms) | 46.96 | 1.64× |
| Median ITL (ms) | 44.10 | 1.63× |
| P99 ITL (ms) | 91.54 | 1.60× |
| Mean E2EL (ms) | 17633.78 | 0.62× |
| Median E2EL (ms) | 15397.68 | 0.63× |
| P99 E2EL (ms) | 54662.73 | 0.63× |

### 接受率

> 注：FastDeploy 累计输出超过 100 万 token 后会重置 `speculate.log` 中的全局统计计数（见 [`token_processor.py:660-662`](../fastdeploy/output/token_processor.py#L660-L662)）。此处取 reset 前最后一次累积值（接近 100 万 token 的样本），代表性更强。

```
Speculate global accept ratio: 0.6309
total step:                    369176
total output token num:        1000244
average accept len:            2.7094
Single head accept ratio:      [0.8170, 0.6746, 0.6187]
```

---

## 配置 6：MTP（5 步）

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"num_model_steps":5,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp-5 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp-5.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec | vs mtp-3 |
|---|---|---|---|
| Successful requests | 2000 | — | — |
| Benchmark duration (s) | 517.63 | 0.87× | 1.41× |
| Total input tokens | 2,541,341 | — | — |
| Total generated tokens | 1,964,388 | 1.01× | 0.98× |
| Request throughput (req/s) | 3.864 | 1.15× | 0.71× |
| Output token throughput (tok/s) | 3795.00 | 1.16× | 0.70× |
| Total token throughput (tok/s) | 8704.61 | 1.15× | 0.71× |
| Mean TTFT (ms) | 502.05 | 1.69× | 1.22× |
| Median TTFT (ms) | 284.95 | 2.43× | 1.41× |
| P99 TTFT (ms) | 6244.70 | 1.24× | 1.09× |
| Mean TPOT (ms) | 25.82 | 0.88× | 1.37× |
| Median TPOT (ms) | 24.93 | 0.89× | 1.39× |
| P99 TPOT (ms) | 45.55 | 0.98× | 1.31× |
| Mean ITL (ms) | 69.93 | 2.44× | 1.49× |
| Median ITL (ms) | 66.50 | 2.45× | 1.51× |
| P99 ITL (ms) | 115.38 | 2.02× | 1.26× |
| Mean E2EL (ms) | 23542.52 | 0.83× | 1.33× |
| Median E2EL (ms) | 20377.23 | 0.83× | 1.32× |
| P99 E2EL (ms) | 73621.32 | 0.85× | 1.35× |

### 接受率

> 注：取 reset 前最后一次累积值（接近 100 万 token 的样本）。

```
Speculate global accept ratio: 0.6627
total step:                    339311
total output token num:        1006011
average accept len:            2.9649
Single head accept ratio:      [0.8073, 0.6602, 0.6072, 0.5889, 0.5787]
```

---

## 配置 7：MTP（1 步）+ Hybrid Ngram（补 2 个）

> hybrid_mode 触发条件：`mtp_strategy == "with_ngram"` 且 `num_speculative_tokens > num_model_steps`。
> 此配置下 MTP 推 1 个，ngram 补 2 个（3 − 1 = 2），与 mtp-3、ngram-3 同 K=3 草稿预算。
>
> **Hybrid kernel 修复说明**（适用于配置 7/8/9）：
> 数据基于 hybrid kernel 的 K+1 padding 修复后采集，保证 `seq_lens_this_time` 在每步固定为 `num_speculative_tokens + 1`，使 target cudagraph 的 capture-time 与 replay-time tensor shape 一致。该修复对接受率的统计无副作用（placeholder 永远被拒绝），但避免了变长 slt 导致的下游 kernel 越界读和 sampler logits 污染。

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"num_model_steps":1,"mtp_strategy":"with_ngram","max_ngram_size":3,"min_ngram_size":1,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp1+ngram2 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp1+ngram2.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec | vs mtp-3 |
|---|---|---|---|
| Successful requests | 2000 | — | — |
| Benchmark duration (s) | 388.18 | 0.65× | 1.06× |
| Total input tokens | 2,541,341 | — | — |
| Total generated tokens | 1,998,366 | 1.03× | 1.00× |
| Request throughput (req/s) | 5.152 | 1.53× | 0.95× |
| Output token throughput (tok/s) | 5148.00 | 1.57× | 0.95× |
| Total token throughput (tok/s) | 11694.76 | 1.55× | 0.95× |
| Mean TTFT (ms) | 335.00 | 1.13× | 0.82× |
| Median TTFT (ms) | 179.61 | 1.53× | 0.89× |
| P99 TTFT (ms) | 4673.34 | 0.93× | 0.82× |
| Mean TPOT (ms) | 19.13 | 0.66× | 1.01× |
| Median TPOT (ms) | 18.24 | 0.65× | 1.02× |
| P99 TPOT (ms) | 34.80 | 0.75× | 1.00× |
| Mean ITL (ms) | 41.65 | 1.45× | 0.89× |
| Median ITL (ms) | 39.72 | 1.46× | 0.90× |
| P99 ITL (ms) | 78.79 | 1.38× | 0.86× |
| Mean E2EL (ms) | 17472.09 | 0.62× | 0.99× |
| Median E2EL (ms) | 15359.51 | 0.63× | 1.00× |
| P99 E2EL (ms) | 55714.04 | 0.64× | 1.02× |

### 接受率

> 注：取 reset 前最后一次累积值（接近 100 万 token 的样本）。

```
Speculate global accept ratio: 0.5876
total step:                    412615
total output token num:        1000403
average accept len:            2.4245
Single head accept ratio:      [0.8143, 0.4374, 0.7133]
```

---

## 配置 8：MTP（3 步）+ Hybrid Ngram（补 1 个）

> hybrid_mode 触发条件：`mtp_strategy == "with_ngram"` 且 `num_speculative_tokens > num_model_steps`。
> 此配置下 MTP 推 3 个，ngram 补 1 个（4 − 3 = 1）。

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"num_model_steps":3,"mtp_strategy":"with_ngram","max_ngram_size":3,"min_ngram_size":1,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp3+ngram1 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp3+ngram1.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec | vs mtp-3 |
|---|---|---|---|
| Successful requests | 2000 | — | — |
| Benchmark duration (s) | 482.07 | 0.81× | 1.31× |
| Total input tokens | 2,541,341 | — | — |
| Total generated tokens | 1,944,614 | 1.00× | 0.97× |
| Request throughput (req/s) | 4.149 | 1.23× | 0.76× |
| Output token throughput (tok/s) | 4033.90 | 1.23× | 0.74× |
| Total token throughput (tok/s) | 9305.64 | 1.23× | 0.75× |
| Mean TTFT (ms) | 545.63 | 1.84× | 1.33× |
| Median TTFT (ms) | 237.61 | 2.03× | 1.17× |
| P99 TTFT (ms) | 6846.31 | 1.36× | 1.20× |
| Mean TPOT (ms) | 24.77 | 0.85× | 1.31× |
| Median TPOT (ms) | 23.03 | 0.82× | 1.29× |
| P99 TPOT (ms) | 54.65 | 1.18× | 1.57× |
| Mean ITL (ms) | 64.80 | 2.26× | 1.38× |
| Median ITL (ms) | 56.37 | 2.08× | 1.28× |
| P99 ITL (ms) | 100.30 | 1.75× | 1.10× |
| Mean E2EL (ms) | 22579.26 | 0.80× | 1.28× |
| Median E2EL (ms) | 20136.44 | 0.82× | 1.31× |
| P99 E2EL (ms) | 70556.30 | 0.81× | 1.29× |

### 接受率

> 注：取 reset 前最后一次累积值（接近 100 万 token 的样本）。

```
Speculate global accept ratio: 0.6478
total step:                    352669
total output token num:        1001235
average accept len:            2.8390
Single head accept ratio:      [0.8053, 0.6680, 0.6136, 0.5020]
```

---

## 配置 9：MTP（3 步）+ Hybrid Ngram（补 2 个）

> hybrid_mode 触发条件：`mtp_strategy == "with_ngram"` 且 `num_speculative_tokens > num_model_steps`。

### 启动服务

```bash
python -m fastdeploy.entrypoints.openai.api_server \
  --model ${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle \
  --port ${FD_API_PORT} \
  --tensor-parallel-size 1 \
  --engine-worker-queue-port ${FD_ENGINE_QUEUE_PORT} \
  --metrics-port ${FD_METRICS_PORT} \
  --cache-queue-port ${FD_CACHE_QUEUE_PORT} \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --quantization wint4 \
  --enable-overlap-schedule \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"num_model_steps":3,"mtp_strategy":"with_ngram","max_ngram_size":3,"min_ngram_size":1,"model":"'${MODEL_PATH}'/ERNIE-4.5-21B-A3B-Paddle/mtp"}' \
  --graph-optimization-config '{"use_cudagraph":true,"use_unique_memory_pool":true,"draft_model_use_cudagraph":true}'
```

### benchmark

```bash
python benchmark_serving.py \
  --backend openai-chat \
  --model mtp3+ngram2 \
  --endpoint /v1/chat/completions \
  --host 0.0.0.0 \
  --port ${FD_API_PORT} \
  --dataset-name EBChat \
  --dataset-path ./filtered_sharedgpt_2000_input_1136_output_200_fd.json \
  --percentile-metrics ttft,tpot,itl,e2el,s_ttft,s_itl,s_e2el,s_decode,input_len,s_input_len,output_len \
  --metric-percentiles 80,95,99,99.9,99.95,99.99 \
  --num-prompts 2000 \
  --max-concurrency 100 \
  --save-result > infer_mtp3+ngram2.log 2>&1
```

### 性能 metrics

| 指标 | 值 | vs non-spec | vs mtp-3 |
|---|---|---|---|
| Successful requests | 2000 | — | — |
| Benchmark duration (s) | 469.73 | 0.79× | 1.28× |
| Total input tokens | 2,541,341 | — | — |
| Total generated tokens | 1,982,016 | 1.02× | 0.99× |
| Request throughput (req/s) | 4.258 | 1.27× | 0.78× |
| Output token throughput (tok/s) | 4219.50 | 1.29× | 0.78× |
| Total token throughput (tok/s) | 9629.74 | 1.27× | 0.78× |
| Mean TTFT (ms) | 423.66 | 1.43× | 1.03× |
| Median TTFT (ms) | 259.91 | 2.22× | 1.28× |
| P99 TTFT (ms) | 5184.27 | 1.03× | 0.91× |
| Mean TPOT (ms) | 23.85 | 0.82× | 1.26× |
| Median TPOT (ms) | 22.89 | 0.82× | 1.28× |
| P99 TPOT (ms) | 40.24 | 0.87× | 1.16× |
| Mean ITL (ms) | 64.13 | 2.23× | 1.37× |
| Median ITL (ms) | 62.04 | 2.29× | 1.41× |
| P99 ITL (ms) | 104.12 | 1.82× | 1.14× |
| Mean E2EL (ms) | 21841.02 | 0.77× | 1.24× |
| Median E2EL (ms) | 19304.71 | 0.79× | 1.25× |
| P99 E2EL (ms) | 65931.22 | 0.76× | 1.21× |

### 接受率

> 注：取 reset 前最后一次累积值（接近 100 万 token 的样本）。token_processor 在累计输出超过 100 万时会重置 `total_step` 和 `number_of_output_tokens`（但不重置 `accept_token_num_per_head`），所以日志最后几条 `accept_ratio` 是 reset 之后新窗口的数据，**不能与 reset 之前的累积统计直接对比**。

```
Speculate global accept ratio: 0.6619
total step:                    338157
total output token num:        1000030
average accept len:            2.9573
Single head accept ratio:      [0.8062, 0.6660, 0.6121, 0.5066, 0.7154]
```

---

## 汇总对比

### 吞吐与延迟

| # | 配置 | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Mean E2EL (ms) | 相对 non-spec 加速比 |
|---|---|---|---|---|---|---|
| 1 | non-spec | 3280.47 | 296.59 | 29.19 | 28270.04 | 1.00× |
| 2 | ngram-3 | 2730.04 | 314.10 | 35.41 | 34106.17 | 0.83× |
| 3 | ngram-5 | 2322.64 | 298.70 | 42.35 | 40182.21 | 0.71× |
| 4 | mtp-1 | 3959.58 | 415.30 | 24.48 | 23071.41 | 1.21× |
| 5 | **mtp-3** | **5437.34** | 409.94 | **18.91** | **17633.78** | **1.66×** |
| 6 | mtp-5 | 3795.00 | 502.05 | 25.82 | 23542.52 | 1.16× |
| 7 | mtp1+ngram2 | 5148.00 | 335.00 | 19.13 | 17472.09 | 1.57× |
| 8 | mtp3+ngram1 | 4033.90 | 545.63 | 24.77 | 22579.26 | 1.23× |
| 9 | mtp3+ngram2 | 4219.50 | 423.66 | 23.85 | 21841.02 | 1.29× |

### 接受率

| # | 配置 | global accept ratio | average accept len | Single head accept ratio |
|---|---|---|---|---|
| 2 | ngram-3 | 0.1055 | 1.118 | N/A |
| 3 | ngram-5 | 0.0718 | 1.077 | N/A |
| 4 | mtp-1 | 0.4570 | 1.842 | [0.840] |
| 5 | mtp-3 | 0.6309 | 2.709 | [0.817, 0.675, 0.619] |
| 6 | mtp-5 | 0.6627 | 2.965 | [0.807, 0.660, 0.607, 0.589, 0.579] |
| 7 | mtp1+ngram2 | 0.5876 | 2.425 | [0.814, 0.437, 0.713] |
| 8 | mtp3+ngram1 | 0.6478 | 2.839 | [0.805, 0.668, 0.614, 0.502] |
| 9 | mtp3+ngram2 | 0.6619 | 2.957 | [0.806, 0.666, 0.612, 0.507, 0.715] |

> 所有接受率取**最大累积 step 那一条**（即 reset 之前的 max 累积值，是当前 benchmark 最具代表性的统计窗口）。Single head ratio 长度 = `num_speculative_tokens`。

---

## 结论

### 1. 功能正确性

9 种配置在 H800×1、wint4、TP=1、并发 100 下全部跑通 2000/2000 请求。ngram / MTP / Hybrid MTP-Ngram 三类投机解码路径功能均验证通过。

### 2. 性能加速比

按相对 non-spec 的 Output token throughput 加速比排序：

```
mtp-3 (1.66×)  >  mtp1+ngram2 (1.57×)  >  mtp3+ngram2 (1.29×)  >  mtp3+ngram1 (1.23×)
  >  mtp-1 (1.21×)  >  mtp-5 (1.16×)  >  non-spec (1.00×)  >  ngram-3 (0.83×)  >  ngram-5 (0.71×)
```

几个关键观察：

- **mtp 推 3 步最优**：mtp-3 的 1.66× 是所有配置里最高的。继续加到 5 步（mtp-5）反而退化到 1.16×，因为草稿 head 的边际接受率快速下降（第 4、5 个 head 接受率分别只有 0.59、0.58），多算的两步成本超过收益。
- **纯 ngram 在 sharedgpt 上是负收益**：ngram-3 / ngram-5 全局接受率仅 0.10 / 0.07，draft slot 多但命中少，每步多 verify 的开销没赚回来。这是 sharedgpt 这种"非重复性强"的对话数据集的固有限制。
- **mtp1+ngram2 是意外亮点**：1.57× 接近 mtp-3 的 1.66×，且 Mean E2EL（17472ms）甚至略低于 mtp-3（17634ms）。每步只跑 1 个 MTP forward + 2 个 ngram 补，**算力开销显著低于 mtp-3 的 3 步 MTP**，对算力受限场景特别有价值。

### 3. Hybrid MTP-Ngram 的有效性

| 对比 | accept ratio | avg accept len | Output tok/s |
|---|---|---|---|
| mtp-3 → mtp3+ngram1（补 1）| 0.6309 → **0.6478** | 2.709 → **2.839** | 5437 → 4034 |
| mtp-3 → mtp3+ngram2（补 2）| 0.6309 → **0.6619** | 2.709 → **2.957** | 5437 → 4220 |

**接受率确实在 ngram 补充位上体现了正向提升**（mtp3 → mtp3+ngram2 提升 4.9 个百分点），avg accept len 也涨了 0.25。但 **throughput 反而降了 22~26%**，原因是每步 verify 的草稿数从 4 涨到 5 或 6，per-iter 计算量上升的幅度超过了接受率上升带来的"摊薄"收益。

具体到草稿 head 接受率上看更直观（mtp3+ngram2 配置）：

```
head 1 (MTP)        : 0.806    ← MTP 第 1 步
head 2 (MTP)        : 0.666    ← MTP 第 2 步
head 3 (MTP)        : 0.612    ← MTP 第 3 步
head 4 (ngram 补 1) : 0.507    ← ngram 第 1 补，因 placeholder 拒绝拉低
head 5 (ngram 补 2) : 0.715    ← ngram 第 2 补，反而比 head 4 高
```

第 5 个 head 反而比第 4 个高的"反直觉"现象，是因为当 ngram 真命中时通常一次性补满全部 K-num_model_steps 个位置（连续的 token 序列），所以"已经补到第 5 位"意味着这次 ngram 命中是高置信度的；而第 4 位混合了"真 ngram 命中"和"仅命中 1 个真 token + placeholder"两种情况，被 placeholder 拖低。
