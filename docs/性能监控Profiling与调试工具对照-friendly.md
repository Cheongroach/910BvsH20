# H20 与 910B4 性能监控、Profiling 和调试工具对照

本文回答三个不同问题：

1. 任务运行数十分钟或数小时后，怎样判断设备是否持续繁忙？
2. 一次请求很慢时，怎样找到时间花在 CPU、运行时还是设备 kernel？
3. 一个 kernel 已经定位后，怎样检查硬件计数器、内存错误或竞争问题？

这三类问题分别对应：

```text
持续监控 → 时间线 Profiling → 单 kernel 分析 / 正确性调试
```

不能用一个工具替代全部三层。Grafana 看到利用率高，不会自动告诉你哪个 kernel 慢；Profiler 找到慢 kernel，也不会自动证明生产任务长期稳定。

本文区分三种证据：

- **本项目真实调用**：2026-10-27 通过 Merlin、Grafana OpenAPI 或硬件任务实际执行；
- **镜像能力检查**：只说明命令或 API 在本次镜像中存在；
- **官方工具能力**：来自 NVIDIA、PyTorch 或 Ascend 官方文档，但不冒充本次任务已经成功采集。

---

## 1. 先区分四个观测层级

| 层级 | H20 主要工具 | 910B4 主要工具 | 输入 | 输出 | 适合回答 |
|---|---|---|---|---|---|
| 平台持续监控 | Grafana、DCGM/Arnold 指标 | Grafana、Ascend/Arnold 指标 | Job/Trial、时间窗 | 低频时序曲线 | 长任务是否持续繁忙、显存是否逼近上限 |
| 节点即时状态 | `nvidia-smi` | `npu-smi` | 设备 ID、刷新周期 | 当前利用率、内存、功耗、温度、版本 | 任务现场是否识别设备、是否有异常卡 |
| 应用时间线 | `torch.profiler`、`nsys` | `torch_npu.profiler`、`msprof` | 应用命令、采集窗口 | CPU/API/kernel 时间线、算子表、trace | 请求慢在 Python、调度、拷贝还是设备执行 |
| kernel 与正确性 | `ncu`、Compute Sanitizer | `msprof` AI Core 指标、msSanitizer | kernel/算子过滤、指标集 | 硬件计数器、流水占比、错误报告 | 一个 kernel 为什么慢，是否越界、竞争或同步错误 |

采样开销通常从上到下增加。长时间持续运行 Grafana 合理；用 `ncu` 或 sanitizer 包住完整 58,240 请求矩阵通常不合理。

---

## 2. 持续监控：两端怎样平行比较

### 2.1 共同入口是 Job 和时间窗

本项目真实执行了：

```bash
merlin-cli --control-plane cn job-v2 runs get-grafana \
  --json '{"job_run_id":"<job_run_id>"}'
```

输入：

- Merlin Job Run ID；
- 控制面，本项目是 `cn`。

输出：

```json
{
  "grafana_url": "<带 Trial、cluster、dc、from、to 的 URL>",
  "trial_id": "<trial_id>"
}
```

这个 URL 已包含历史任务的准确时间窗。不要手工写 `now-30m` 查询一周前的任务。

下一步先列面板：

```bash
merlin-cli --control-plane cn grafana describe \
  --json '{"url":"<grafana_url>"}'
```

输出是面板 ID、标题、类型和底层查询。最后按面板 ID 拉结构化时序：

```bash
merlin-cli --control-plane cn grafana panel-data \
  --json '{"url":"<grafana_url>","panel_id":7}'
```

关键输出是：

```text
metric：底层指标名
tags：设备、Pod 等过滤信息
dps：timestamp → value
```

**解释时以 `metric` 为准，不以面板标题为准。**

### 2.2 本次真实查询的共同指标

数据来自 2026-10-19 的 Qwen3.5-4B 历史任务。2026-10-27 重新调用 Grafana OpenAPI 获得时序。完整脱敏汇总见：

```text
results/observability/monitoring-summary.json
```

| 指标 | H20 | 910B4 | 怎样解释 |
|---|---:|---:|---|
| DutyCycle 样本数 | 41 | 54 | 两个任务时间窗不同，不能把样本当配对重复 |
| DutyCycle 平均/峰值 | 36.4% / 96% | 38.5% / 100% | 窗口含初始化、运行和空闲；不是稳态 kernel 利用率 |
| 设备内存总量 | 102,625,181,696 bytes | 面板原值 65,536 | 两端原生单位不同；910B4 原值对应 64 GiB 资源，不直接相除 |
| 设备内存已用峰值 | 91,086,716,928 bytes | 未返回 used series | “没有序列”不等于使用量为零 |
| 内存带宽利用率平均/峰值 | 7.83% / 44.99% | 14.48% / 57% | 两端硬件计数器定义不同，不写成带宽性能倍率 |
| 功耗平均/峰值 | 204.4 W / 397.0 W | 155.4 W / 315.7 W | 没有按有效 token 对齐，不能算能效排名 |

DutyCycle 的平均值低于峰值，说明完整任务窗口中存在低活动段；初始化、等待或任务结束后的空闲都可能贡献这些样本。它不能推翻吞吐实验，也不能证明设备 kernel 只用了三成能力。

### 2.3 名字相似但不能平行解释的指标

H20 返回了三条不同的设备计数器：

| H20 指标 | 平均 | 峰值 |
|---|---:|---:|
| `gpu_tensor_activity` | 19.45% | 87.41% |
| `gpu_sm_activity` | 26.01% | 97.54% |
| `gpu_sm_occupancy` | 5.27% | 20.50% |

910B4 看板也显示 “NPU Tensor Core Activity” 和 “NPU SM Activity”，但真实查询返回：

```text
lab.arnold.instance_monitor.gpu_tensor_activity
lab.arnold.instance_monitor.gpu_sm_activity
```

两条时序的样本数、最小值、最大值和平均值完全相同：

```text
54 points，0% → 70%，平均 23.7761%
```

本项目因此不把它们解释成两个独立的 NPU 硬件计数器。正确结论是：

> 当前看板给出了两个标题，但底层指标命名和完全相同的数值不足以证明它们分别测量了 910B4 的两类独立执行单元。

需要确认 AI Core、Vector Core、Cube 或搬运单元的占比时，应使用 `msprof --aic-metrics=PipeUtilization`，而不是从这两个 Grafana 标题反推。

### 2.4 H20 看板特有的视角

本次 H20 看板额外暴露：

| 面板 | 输入 | 输出 | 本次结果 | 使用时机 |
|---|---|---|---|---|
| SM Occupancy | Trial + 时间窗 | `gpu_sm_occupancy` | 平均 5.27%，峰值 20.50% | 判断驻留 warp/资源限制线索 |
| FP engine activity | 同上 | FP16/FP32/FP64 engine series | FP32 峰值 1.31%，FP64 为 0 | 判断普通 FP 管线活动；不等于 Tensor Core 活动 |
| NVLink bandwidth | 同上 | RX/TX bytes/s | 两条均为 0 | 单卡任务没有观察到 NVLink 流量 |
| PCIe bandwidth | 同上 | RX/TX bytes/s | 返回非零时序 | 检查主机与设备传输阶段 |

“当前 NPU 看板没有 SM Occupancy”只说明这套看板没有同名、同语义面板，不表示 910B4 硬件无法观测核心利用率。

### 2.5 910B4 看板特有的视角

本次 910B4 看板额外暴露：

| 面板 | 输入 | 输出 | 本次结果 | 使用时机 |
|---|---|---|---|---|
| HCCS RX/TX | Trial + 时间窗 | 芯片互联带宽时序 | 单卡任务返回 0 series | 多卡 HCCL 通信慢时检查 |
| ECC 单/双 bit 错误 | 同上 | 累计错误计数 | 均为 0 | 排查硬件可靠性异常 |
| ECC isolated pages | 同上 | 隔离页计数 | 面板存在 | 检查错误是否导致页隔离 |
| PCIe posted/non-posted/completion | 同上 | 分事务类型带宽 | 本次查询的 posted RX 为 0 | 深分 PCIe 事务流量 |
| HBM frequency | 同上 | 频率时序 | 返回全 0 | 不能区分“不活动”和“采集不支持” |

910B4 的 HCCS 与 H20 的 NVLink 都是设备互联视角，但协议、拓扑和计数器不同，只能做概念对应，不能比较原始数值。

### 2.6 零序列、空序列和零利用率是三件事

本次查询出现三种返回：

1. **非零时序**：例如 H20 DutyCycle；
2. **全零时序**：例如本次单卡 HCCS；
3. **没有时序**：例如 910B4 历史任务的 CPU 面板和 HBM used。

对应解释：

- 非零时序可以计算本窗口的最小、最大和平均；
- 全零可能是没有活动，也可能是采集器不支持，需结合工具文档和负载判断；
- 空序列只说明当前查询没有数据，不能写成资源使用为零。

---

## 3. 节点即时状态：`nvidia-smi` 与 `npu-smi`

| 项目 | H20 | 910B4 |
|---|---|---|
| 工具 | `nvidia-smi` | `npu-smi` |
| 最小输入 | 无参数或 GPU ID | `info`、NPU ID、chip ID |
| 主要输出 | 型号、驱动、显存、利用率、功耗、温度、进程 | 型号、驱动/固件、HBM、AI Core、功耗、温度、健康状态 |
| 使用时机 | 任务刚启动、设备不可见、显存异常、降频 | 任务刚启动、设备/芯片不可见、健康告警、HBM 异常 |
| 不适合 | 解释单个 kernel 内部瓶颈 | 解释单个算子内部流水瓶颈 |

典型命令：

```bash
# H20
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version \
  --format=csv,noheader

# 910B4
npu-smi info
npu-smi info -l
npu-smi info watch -i <device_id> -c <chip_id> \
  -d <seconds> -s <watch_type>
```

这类命令的输出是现场快照。需要观察 20 分钟内是否周期性掉速时，使用 Grafana；需要知道某个矩阵乘 kernel 为什么慢时，使用 profiler。

---

## 4. 框架级 Profiling：最接近的一组平行工具

### 4.1 `torch.profiler` 与 `torch_npu.profiler`

| 项目 | H20 | 910B4 |
|---|---|---|
| Python API | `torch.profiler.profile` | `torch_npu.profiler.profile` |
| Device activity | `ProfilerActivity.CUDA` | `ProfilerActivity.NPU` |
| 共同输入 | activities、采集代码段、shape/memory/stack 开关 | 同左 |
| 共同输出 | 算子聚合表、CPU/device 时间、Chrome trace | 算子聚合表、CPU/NPU 时间、trace 与 Ascend 分析文件 |
| 适合 | 从 Python op 定位到 CUDA kernel | 从 PyTorch op 定位到 NPU 算子 |
| 主要限制 | 采集会增加开销，长任务应使用 schedule | 同左；可用参数依赖 torch-npu/CANN 版本 |

仓库提供同一输入探针：

```bash
python src/profile_probe.py \
  --device cuda \
  --matrix-size 2048 \
  --warmup 3 \
  --steps 10 \
  --output-dir /tmp/profile-probe
```

```bash
python src/profile_probe.py \
  --device npu \
  --matrix-size 2048 \
  --warmup 3 \
  --steps 10 \
  --output-dir /tmp/profile-probe
```

固定输入：

```text
X, Y: [2048, 2048]
dtype: BF16
计算: SiLU(X @ Y)
warmup: 3
profiled steps: 10
```

输出：

| H20 输出 | 910B4 输出 |
|---|---|
| `cuda-torch-profiler.json` | `trace_view.json` |
| `cuda-summary.json` | `kernel_details.csv`、`operator_details.csv` |
| 按 Self CUDA time 排序的算子表 | memory、task、step CSV 和分析数据库 |

两端都输出版本、设备、shape 和文件大小摘要，但原生 trace 结构不同。

接口定义见 [PyTorch Profiler](#ref-1) 和 [Ascend PyTorch Profiler](#ref-5)。

使用时机：

- 已知一个请求慢，但不知道慢在 Python、框架调度还是设备算子；
- 需要把 `aten::matmul`、vLLM op 与实际设备活动关联；
- 比较编译前后算子数量、shape 或内存分配是否变化。

不要直接把 profiler 中的单次时间当生产 P95。Profiler 改变执行环境，目标是定位，不是替代正式 benchmark。

---

## 5. 应用时间线：`nsys` 与 `msprof`

### 5.1 H20：Nsight Systems

输入是“待运行应用 + 采集范围”。下面是官方命令形状；本次 H20 镜像没有安装 `nsys`，因此没有生成 `.nsys-rep`：

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=/tmp/h20-nsys \
  python src/profile_probe.py \
    --device cuda --matrix-size 1024 \
    --warmup 2 --steps 5 --workload-only
```

输出：

```text
h20-nsys.nsys-rep
```

可继续执行：

```bash
nsys stats h20-nsys.nsys-rep
nsys export --type sqlite h20-nsys.nsys-rep
```

主要观察：

- CPU 线程何时提交 CUDA API；
- kernel 与 memcpy 在各 stream 上何时执行；
- kernel 之间是否有空洞；
- NVTX 标记的 prefill/decode 是否与设备时间对齐。

命令与采集范围选项见 [Nsight Systems User Guide](#ref-2)。

### 5.2 910B4：CANN msprof

输入同样是“待运行应用 + 采集项”：

```bash
msprof \
  --application="python src/profile_probe.py \
    --device npu --matrix-size 1024 \
    --warmup 2 --steps 5 --workload-only" \
  --output=/tmp/npu-msprof \
  --aic-mode=task-based \
  --aic-metrics=PipeUtilization
```

官方文档还支持按需打开：

```text
AscendCL/runtime API
task time
AI CPU
AI Core / Vector Core
系统 CPU 和内存
HCCS / PCIe
```

输出不是 `.nsys-rep`，而是 `PROF_*` 目录及解析后的 JSON/CSV，例如 AI Core utilization、HCCS、PCIe、CPU 或内存文件。

主要观察：

- Host API、runtime task、stream 和 NPU 算子的时间关系；
- AI Core 与 AI Vector Core 活动；
- task-based 或 sample-based 采样；
- `PipeUtilization` 中计算与搬运流水的耗时占比。

采集项与输出文件见 [Ascend msprof 系统数据采集](#ref-6) 和 [PipeUtilization](#ref-8)。

### 5.3 两者相似，但不能交换报告

| 共同点 | 关键差异 |
|---|---|
| 都能包住一个应用并采集时间线 | `.nsys-rep` 与 `PROF_*`/CSV 格式不兼容 |
| 都应限制采集窗口 | `nsys` 常用 CUDA/NVTX 语义，`msprof` 使用 AscendCL/task/AI Core 语义 |
| 都会增加运行开销和数据量 | 采集项名称和硬件计数器不同 |
| 都可先看系统时间线，再下钻 kernel/算子 | 不能用一个工具打开另一端的原生报告 |

---

## 6. 单 kernel 深挖：不存在严格一对一映射

### 6.1 H20：Nsight Compute `ncu`

示例输入如下。本次 H20 镜像没有安装 `ncu`，因此这条命令是复现实验所需的镜像准备说明，不是本次已生成的报告：

```bash
ncu \
  --set basic \
  --target-processes all \
  --output /tmp/h20-ncu \
  python src/profile_probe.py \
    --device cuda --matrix-size 1024 \
    --warmup 2 --steps 3 --workload-only
```

输出：

```text
h20-ncu.ncu-rep
```

`ncu` 可以按 kernel 名过滤，并采集 SM throughput、occupancy、内存访问、warp stall 等 NVIDIA 指标。部分指标需要 replay kernel 多次，因此开销可能远大于 `nsys`。

使用时机：

- 时间线已经定位到一个 CUDA kernel；
- 需要判断是计算吞吐、occupancy、内存访问还是 stall 限制；
- kernel 足够确定、输入可重复。

不应先用 `ncu` 包住完整在线服务。

Kernel replay、指标集和 `.ncu-rep` 见 [Nsight Compute CLI](#ref-3)。

### 6.2 910B4：`msprof --aic-metrics` 与 `msprof op`

Ascend 侧没有与 `.ncu-rep`、SM metric namespace 完全兼容的工具。

最接近的下钻方式是：

```text
msprof --aic-metrics=PipeUtilization
msprof --aic-metrics=Memory
msprof --aic-metrics=MemoryL0
msprof --aic-metrics=MemoryUB
msprof --aic-metrics=ResourceConflictRatio
msprof --aic-metrics=L2Cache
```

算子开发场景还可使用 `msprof op`，输出计算、搬运单元占比、内存热力图或 Roofline 等数据。

使用时机：

- 时间线已定位到一个 Ascend 算子；
- 需要区分 Cube/Vector 计算与 MTE 搬运；
- 需要看 UB、L0、L2 或资源冲突。

因此，正确的平行关系是：

```text
ncu：NVIDIA kernel 硬件计数器和 replay
msprof/msprof op：Ascend task/算子与 AI Core 流水指标
```

它们回答相似层级的问题，但不是同一套指标。

---

## 7. 正确性调试：Compute Sanitizer 与 msSanitizer

### 7.1 H20：Compute Sanitizer

输入：

```bash
compute-sanitizer --tool memcheck <application>
compute-sanitizer --tool racecheck <application>
compute-sanitizer --tool initcheck <application>
compute-sanitizer --tool synccheck <application>
```

输出是错误类型、kernel、地址、线程/块和调用栈等文本报告。

使用时机：

- CUDA illegal memory access；
- shared memory race；
- 未初始化 global memory；
- 错误同步原语。

它是正确性工具，不是性能 benchmark。

四类检查的官方定义见 [Compute Sanitizer](#ref-4)。

### 7.2 910B4：msSanitizer

输入：

```bash
mssanitizer --tool=memcheck <application>
mssanitizer --tool=memcheck --leak-check=yes <application>
```

官方文档列出的输出包括：

- GM/UB/L0/L1 非法读写；
- 多核踩踏；
- 非对齐访问；
- 非法释放；
- 内存泄漏；
- 已分配但未使用内存。

若编译时启用 sanitizer 和调试信息，报告可包含算子源文件、行号和调用栈。

重要限制：官方文档说明 msSanitizer 不支持对 Ascend Transformer Boost 算子仓做内存检测。对本项目的融合算子或第三方算子，必须先确认它是否位于支持范围。

支持的异常类型与限制见 [Ascend msSanitizer 内存检测](#ref-9)。

### 7.3 相似点和不可互换之处

| 维度 | Compute Sanitizer | msSanitizer |
|---|---|---|
| 共同目标 | 设备内存错误和并行正确性 | 设备内存错误和多核正确性 |
| 设备内存术语 | global/shared/local 等 CUDA 空间 | GM、UB、L0、L1 等 Ascend 空间 |
| 竞争检查 | shared-memory racecheck | 多核踩踏/竞争检测 |
| 同步检查 | 独立 `synccheck` | 按 Ascend 算子执行模型检查 |
| 报告能否互读 | 否 | 否 |
| 使用代价 | 高，应缩小到最小复现 | 高，应缩小到目标算子 |

---

## 8. 哪些方法只在一侧成立

### 8.1 NVIDIA 侧特有或本次只在 H20 侧验证

1. **CUDA/NVTX 时间线**：`nsys` 直接理解 CUDA API、stream、CUDA Graph 和 NVTX。
2. **Nsight Compute kernel replay**：`ncu` 可为同一 CUDA kernel 分多次 replay 采集指标。
3. **SM/warp 指标体系**：occupancy、warp stall、SASS/source correlation 属于 NVIDIA 执行模型。
4. **Compute Sanitizer 四工具划分**：memcheck、racecheck、initcheck、synccheck 是 CUDA 专用入口。
5. **本次 Grafana 的独立 SM/Tensor/occupancy 曲线**：真实返回了三条不同计数器。

不能把这些名称直接移植到 910B4。需要把问题翻译为 AI Core/Vector/Cube/MTE、task 和 stream。

### 8.2 Ascend 侧特有或本次只在 910B4 侧暴露

1. **PipeUtilization**：按 AI Core 计算和 MTE 搬运流水拆解耗时占比。
2. **task-based / sample-based AI Core 采集**：由 `--aic-mode` 选择。
3. **HCCS 系统采集**：`--sys-interconnection-profiling` 可采 HCCS 与 PCIe。
4. **GM/UB/L0/L1 内存异常术语**：msSanitizer 按 Ascend 内存层级报告。
5. **本次 Grafana 的 ECC 隔离页和细分 PCIe transaction 面板**。

“H20 看板没有 ECC isolated-pages 面板”只表示当前 dashboard 没有公开该面板，不表示 NVIDIA 设备没有 ECC 或其他健康监控。

---

## 9. 本次真实调用状态

### 9.1 历史监控调用

| 项目 | H20 | 910B4 |
|---|---|---|
| `get-grafana` | 成功 | 成功 |
| `grafana describe` | 成功，列出 GPU/CPU/Memory/Network 等面板 | 成功，列出 NPU/HBM/HCCS/ECC 等面板 |
| `panel-data` DutyCycle | 41 个点 | 54 个点 |
| `panel-data` device memory | total + used | total；used 未返回 |
| `panel-data` bandwidth/power | 成功 | 成功 |
| 历史 merlin-profiling 资产 | 空 | 空 |

历史资产为空意味着 2026-10-19 任务结束后不能凭空生成 kernel 时间线。Grafana 可以事后查询，是因为平台持续采集；Profiler 必须在应用运行时附加或包住应用。

### 9.2 新 profiler probe

2026-10-27 在同一资源组完成两端单卡 probe。脱敏机读结果见：

```text
results/observability/profiling-summary.json
```

共同输入是 BF16 `X,Y∈R^(2048×2048)`，执行 3 次预热和 10 次 `SiLU(X @ Y)`。

| 项目 | H20 | 910B4 |
|---|---|---|
| 框架 profiler | `torch.profiler`，返回码 0 | `torch_npu.profiler`，返回码 0 |
| 框架输出 | 82,151-byte Chrome trace JSON | 536,119 bytes、88 个文件 |
| 代表性文件 | `cuda-torch-profiler.json` | `trace_view.json`、`kernel_details.csv`、`operator_details.csv`、memory CSV、分析数据库 |
| 系统/应用 profiler | `nsys` 未安装 | `msprof` 返回码 0 |
| kernel profiler | `ncu` 未安装 | `msprof --aic-metrics=PipeUtilization` 成功 |
| 正确性工具 | Compute Sanitizer 2025.2.1.0 存在 | `mssanitizer` 存在 |
| 任务终态 | Job done、Trial finished、Pod exit 0 | Job done、Trial finished、Pod exit 0 |

H20 算子表显示，10 次 `2048×2048` BF16 计算的 Self CUDA 总计 1.553 ms：

```text
aten::mm    1.417 ms，91.22%
aten::silu  0.136 ms， 8.78%
```

这些数值只验证 profiler 能把框架 op 与 CUDA 时间关联。它们不包含 vLLM、模型层、KV cache 或请求调度。

910B4 的框架 profiler 生成：

```text
trace_view.json               158,906 bytes
ascend_pytorch_profiler.db    188,416 bytes
kernel_details.csv              1,763 bytes
operator_details.csv            3,026 bytes
task_time.csv                   2,196 bytes
```

外层 `msprof` 另生成 `msprof_*.json`、`op_summary_*.csv`、`task_time_*.csv` 和数据库，证明应用包装式采集路径可用。

### 9.3 真实调用暴露的 API 与镜像差异

第一版通用 probe 在 910B4 上完成 CANN 采集和解析后，调用：

```python
prof.key_averages()
```

触发 `AttributeError`。`torch_npu.profiler.profile` 不提供 CUDA 路径使用的这个后处理方法。修正版改用 `tensorboard_trace_handler(..., analyse_flag=True)`，从生成的 CSV/JSON/数据库读取结果后通过。

这次还观察到：

- H20 镜像有 `compute-sanitizer` 和旧的 `nvprof`，没有 `nsys`/`ncu`；
- 910B4 镜像有 `msprof` 与 `mssanitizer`，没有独立 `msprof-analyze` 命令；
- 910B4 probe 实际加载 `/usr/local/Ascend/cann-8.5.1`，不是历史 benchmark 的 CANN 9.1；
- 两端任务都生成本地 trace，但没有上传至 Merlin profiling 资产服务，所以资产查询仍为空。

因此，“工具由生态提供”“工具安装在当前镜像”“本次调用成功”“产物已持久化”是四项独立验收。

---

## 10. 推荐排查顺序

### 场景 A：线上吞吐突然下降

```text
1. Grafana 对齐下降时间窗
2. 同时看请求吞吐、DutyCycle、设备内存、CPU、PCIe
3. DutyCycle 低：
   查排队、CPU、数据、调度和同步空洞
4. DutyCycle 高但吞吐低：
   用 torch profiler + nsys/msprof 采一个短窗口
5. 定位到单 kernel/算子：
   H20 用 ncu；910B4 用 msprof AI Core 指标
```

### 场景 B：设备报 illegal memory access 或算子异常

```text
1. 缩成最小 shape 和单算子复现
2. H20：Compute Sanitizer
3. 910B4：msSanitizer
4. 保留编译调试信息
5. 修复后重新做数值测试，再做性能测试
```

### 场景 C：高并发 27B 退化

当前项目最需要的证据是：

```text
客户端请求时间线
vLLM scheduler / waiting / running 序列
KV cache 使用与抢占事件
CPU/runtime/device kernel 时间线
AI Core 或 SM 的短窗口计数器
```

仅有任务级 DutyCycle 平均值不足以在 scheduler、KV cache、图编译和 kernel 之间定位根因。

---

## 11. 不能从这些工具直接得出的结论

- Grafana 平均利用率不能替代相同请求下的吞吐和 P95。
- 面板标题相同不能证明底层硬件计数器相同。
- `command -v` 找到 profiler 不能证明有权限完成采集。
- Profiler trace 成功生成不能证明模型输出正确。
- `ncu` 与 `msprof` 的百分比不能直接相除。
- 单卡 NVLink/HCCS 为零不能代表多卡通信性能。
- 一个时间窗的平均功耗不能直接代表每百万 token 的能耗。
- 空时序不能写成零使用量。

---

## 参考资料

<a id="ref-1"></a>
1. [PyTorch Profiler](https://docs.pytorch.org/docs/main/profiler.html)

<a id="ref-2"></a>
2. [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

<a id="ref-3"></a>
3. [NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)

<a id="ref-4"></a>
4. [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

<a id="ref-5"></a>
5. [Ascend PyTorch Profiler 数据说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC2alpha002/devaids/auxiliarydevtool/atlasprofiling_16_0038.html)

<a id="ref-6"></a>
6. [Ascend msprof 系统数据采集](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/Profiling/atlasprofiling_16_0012.html)

<a id="ref-7"></a>
7. [Ascend msprof AI 任务采集示例](https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/atlasprofiling_16_0011.html)

<a id="ref-8"></a>
8. [Ascend msProf PipeUtilization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0108.html)

<a id="ref-9"></a>
9. [Ascend msSanitizer 内存检测](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/opdev/optool/atlasopdev_16_0042.html)

<a id="ref-10"></a>
10. [Ascend npu-smi 命令参考](https://www.hiascend.com/doc_center/source/zh/Atlas%20200I%20A2/23.0.RC3/re/npu/Ascend%20310B%2023.0.RC3%20npu-smi%20%E5%91%BD%E4%BB%A4%E5%8F%82%E8%80%83.pdf)
