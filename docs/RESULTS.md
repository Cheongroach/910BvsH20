# Results and conclusions

## Executive summary

HeteroInferLab compares complete single-accelerator inference deployments:
NVIDIA H20 and Huawei Ascend 910B4-64G. It does not attempt to isolate the
silicon from the runtime, compiler, kernels, and vLLM backend.

The published data contains:

- 480 unique cells in the primary throughput matrix
- 1,326 underlying throughput measurements after repeats and ablations
- 728 request-level latency cells
- 58,240 successful latency requests with zero recorded request errors
- 23,520 paired closed-loop request inputs with zero hash mismatches

## Selected generation result

The following uses the online closed-loop path, combo configuration,
concurrency 32, 1,024 input tokens, and 128 generated tokens.

| Model | H20 output tok/s | 910B4 output tok/s | H20 / 910B4 |
|---|---:|---:|---:|
| Qwen3.5-2B | 3,189.35 | 1,567.26 | 2.03x |
| Qwen3.5-4B | 1,413.03 | 757.44 | 1.87x |

This selected shape favors H20, but it is not a universal hardware ranking.
For example, in the same online matrix at concurrency 32 and 1,024 input
tokens, 910B4 was faster for Qwen3-14B with one output token and also slightly
faster with 64 output tokens. Prompt length, output length, model architecture,
batching, and runtime implementation all change the result.

![Generation throughput](../results/latency/charts/generation-throughput.png)

## The 27B concurrency regression

For all three same-size model revisions on the tested 910B4 stack, moving from
closed-loop concurrency 8 to 32 at the 1,024-to-128 shape reduced output
throughput from about 97-98 tok/s to about 49 tok/s. E2E P95 rose from about
10.3 seconds to about 102 seconds.

The corresponding H20 runs increased from about 181-182 tok/s to 255-257
tok/s, although their E2E P95 also rose to about 22 seconds. Seeing nearly the
same 910B4 behavior on Qwen3.5, Qwen3.6, and Qwen3.8-27B makes a single model
revision a weak explanation.

This result demonstrates a deployment-level concurrency limit or regression.
The available experiment does not prove whether the root cause is scheduling,
memory pressure, graph behavior, kernel performance, or another backend
component. Profiler and runtime traces would be required for that claim.

## How to read the data

- `requests_s` measures completed requests divided by the cell wall time.
- `output_tokens_s` is the main generation-throughput metric.
- `ttft_s_p95` measures tail time to first token.
- `tpot_s_p95` measures per-request tail time per output token after the first.
- `itl_s_p95` uses only observable one-token stream intervals. Merged stream
  chunks reduce exact ITL coverage.
- `e2e_s_p95` is the request completion tail latency.

Throughput should always be compared at the same model, input length, output
length, configuration, mode, arrival policy, and concurrency.

## Limitations

1. The two devices used different software stacks, so results represent the
   full deployments rather than chip-only capability.
2. HTTP tests used worker-local loopback and do not include an external
   gateway or production network.
3. Open-loop arrival rates were derived independently from each deployment's
   measured closed-loop window. Those rows compare stability at relative load,
   not equal absolute offered QPS.
4. Fixed-length generation controls compute shape but is not an answer-quality
   benchmark.
5. Per-cell sample counts are bounded and do not provide independent
   confidence intervals for every P95 estimate.
6. Aggregate CSVs are public; raw request traces and scheduler logs are
   omitted because they contained environment metadata and were too large for
   a practical source repository.
