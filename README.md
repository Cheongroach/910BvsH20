# HeteroInferLab

HeteroInferLab 是一套 H20 与 Ascend 910B4 单卡大模型推理对比实验。它要回答的不是“哪张卡绝对更强”，而是：

> 在模型、输入长度、输出长度和并发都相同时，两套完整推理环境的吞吐和延迟分别怎样？

这里比较的是**硬件、运行时、编译器、算子和 vLLM 后端组成的完整部署栈**，不是脱离软件环境的芯片峰值算力。

## 先看结论

在本次实测环境中：

- Qwen3.5-2B/4B 的长文本生成更适合 H20。固定为 1,024 输入 token、128 输出 token、在线并发 32 和组合优化时，H20 的生成吞吐分别是 910B4 的 **2.03 倍**和 **1.87 倍**。
- 这个结论不能推广到所有任务。9B/14B 的单 token 判断和首 token 延迟中，910B4 在部分配置下更快。
- 27B 模型在所测 910B4 部署栈上出现明显的高并发退化：并发从 8 增至 32 后，生成吞吐约减半，95% 请求的完整响应时间（E2E P95）从约 10 秒升至约 102 秒。
- 最后一项是**实验现象**，不是已经定位的芯片根因。现有数据无法区分调度、缓存、编译图或算子的责任。

下图只展示一个固定场景：在线闭环、并发 32、1,024 输入 token、128 输出 token、组合优化。柱越高，单位时间生成的 token 越多。

![固定场景下的生成吞吐](results/latency/charts/generation-throughput.png)

完整解释见 [实验结果](docs/RESULTS.md)。

## 建议阅读顺序

第一次阅读不需要先看代码或 CSV：

1. [实验结果](docs/RESULTS.md)：知道哪些结论成立，哪些不成立。
2. [实验方法](docs/METHODOLOGY.md)：理解“480 格”、TTFT、TPOT、P95 等概念。
3. [STAR 与简历写法](docs/STAR.md)：理解这个项目能证明哪些工程能力。
4. [项目过程友好版](docs/CONVERSATION_HISTORY-friendly.md)：了解需求如何演变、哪里踩过坑、如何纠错。
5. [完整脱敏对话](docs/CONVERSATION_HISTORY.md)：用于审计，不建议从头阅读。

## “480 格”是什么

一“格”就是一种唯一实验配置。例如：

```text
H20 × Qwen3.5-4B × combo
× 1,024 输入 token × 128 输出 token × batch 32
```

首轮吞吐实验的完整组合是：

```text
2 种硬件
× 5 个模型
× 2 种配置
× 2 种输入长度
× 3 种输出长度
× 4 种 batch
= 480 格
```

“格”不是请求数，也不是调度任务数。一格可以包含多条请求和多次重复测量，一个调度任务也可以连续运行多格。

项目最终公开了两套数据：

| 数据集 | 回答的问题 | 规模 |
|---|---|---:|
| 吞吐矩阵 | 饱和负载下，每秒能处理多少请求或 token？ | 480 格，1,326 次测量 |
| 请求级延迟矩阵 | 用户要等多久才能看到首 token、完整答案？ | 728 格，58,240 个成功请求 |

## 四个最重要的指标

| 指标 | 直观含义 | 趋势 |
|---|---|---|
| Output tokens/s | 每秒生成多少 token，衡量生成吞吐 | 越高越好 |
| TTFT | 从请求发出到看到第一个 token | 越低越好 |
| TPOT | 首 token 之后，平均生成一个 token 要多久 | 越低越好 |
| E2E P95 | 95% 的请求能在多长时间内完成 | 越低越好 |

吞吐高不等于单个用户等待时间短。提高并发通常能增加总吞吐，也可能让 TTFT 和 P95 变差，因此这些指标必须一起看。

## 仓库内容

```text
src/
  throughput_benchmark.py   固定输入/输出形状的吞吐与质量探针
  latency_benchmark.py      离线异步与在线流式延迟测试
analysis/
  validate_public_results.py
  plot_public_results.py
data/
  manifest.json             吞吐实验的模型文件清单与哈希
  latency_model_manifest.json
  workload.jsonl            512 条选定的 SQuAD v1.1 样本
results/
  throughput/               吞吐聚合数据与图
  latency/                  请求级延迟聚合数据与图
docs/
  RESULTS.md                结论、反例与适用范围
  METHODOLOGY.md            实验设计和指标定义
  STAR.md                   简历和面试表达
  CONVERSATION_HISTORY-friendly.md
  CONVERSATION_HISTORY.md   完整脱敏对话档案
```

## 验证公开结果

安装绘图和校验所需的轻量依赖：

```bash
python -m pip install -r requirements.txt
```

重新检查矩阵完整性和请求数量：

```bash
python analysis/validate_public_results.py
```

重新生成 README 中的主图：

```bash
python analysis/plot_public_results.py
```

预期校验结果包括：

- 吞吐主矩阵 `480/480` 格；
- 延迟矩阵 `728/728` 格；
- 58,240 个请求全部成功；
- 23,520 对跨硬件输入哈希一致。

## 运行新的实验

基准脚本还需要 `transformers`、`aiohttp`，以及对应硬件可用的 vLLM 环境。NVIDIA 与 Ascend 的运行时依赖不同，因此没有放进同一个跨平台 requirements 文件。

准备如下目录：

```text
/tmp/ib/
  manifest.json
  workload.jsonl
  2b/
  4b/
  9b/
  14b/
  q3-4b/
```

运行吞吐实验：

```bash
python src/throughput_benchmark.py \
  --device cuda \
  --root /tmp/ib \
  --model 4b \
  --config baseline \
  --suite sweep
```

运行离线请求级延迟：

```bash
python src/latency_benchmark.py \
  --device cuda \
  --root /tmp/ib \
  --model 4b \
  --config combo \
  --mode offline
```

运行在线流式延迟：

```bash
python src/latency_benchmark.py \
  --device cuda \
  --root /tmp/ib \
  --model 4b \
  --config combo \
  --mode online \
  --url http://127.0.0.1:18081
```

## 不能从本项目得出的结论

- 不能说 H20 在所有模型和任务上都比 910B4 快。
- 不能把本机 loopback HTTP 的 P95 当作生产服务 SLA。
- 不能把固定输出长度实验当作自然答案质量评估。
- 不能用当前数据证明 27B 退化的具体根因。
- 不能把不同绝对到达率的开放负载直接当作公平硬件对比。

模型权重不在本仓库中。SQuAD v1.1 数据保留原始数据集归属说明。
