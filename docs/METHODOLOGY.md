# Methodology

## Goal

The benchmark asks a deployment question: under controlled request shapes,
how do single-device H20 and Ascend 910B4 vLLM stacks behave in throughput,
first-token latency, generation latency, and tail latency?

It is designed for repeatable comparison, not for estimating theoretical peak
FLOPS or declaring one accelerator universally faster.

## Reproducibility controls

- Model files are identified by revision and SHA-256 in the manifests.
- The selected 512-example SQuAD v1.1 workload has a pinned SHA-256.
- Prompt token counts are exact after tokenization.
- Generation uses temperature 0, fixed minimum and maximum output lengths,
  EOS suppression, and a fixed seed.
- Non-prefix-cache cells receive a unique leading document identifier.
- Full-shape warmups run before formal timing.
- Cross-hardware closed-loop requests are paired using input hashes.
- Engine versions and effective startup arguments are emitted with each run.

Model weights are not stored in this repository. The manifests describe the
artifacts needed to reconstruct the local model directories.

## Primary throughput matrix

The 480 cells are the Cartesian product:

```text
2 hardware targets
x 5 models
x 2 configurations
x 2 input lengths
x 3 output lengths
x 4 batch sizes
= 480 unique cells
```

The dimensions are:

- Hardware: H20, 910B4
- Models: Qwen3.5-2B/4B/9B, Qwen3-14B, Qwen3-4B
- Configuration: baseline, combo
- Input length: 1,024 or 2,048 tokens
- Output length: 1, 64, or 128 tokens
- Batch size: 1, 8, 32, or 128

A cell is a unique configuration, not a request and not a scheduler job.
Repeated runs and follow-up ablations explain why the dataset contains more
measurements than primary cells.

`baseline` uses eager execution without chunked prefill. `combo` enables the
available graph path and chunked prefill. Additional rows independently probe
graph/compile modes, chunked prefill, prefix caching, FP8, and MTP where the
software stack supports them.

## Request-level latency matrix

The latency benchmark adds Qwen3.5, Qwen3.6, and Qwen3.8-27B models and
exercises two delivery paths:

- Offline: asynchronous vLLM token delivery
- Online: OpenAI-compatible SSE over an isolated loopback endpoint

Closed-loop tests use concurrency 1, 8, or 32, depending on the selected
matrix. Open-loop tests use Poisson arrivals at deployment-specific fractions
of a measured closed-loop service window.

Each request records token arrival times, token counts, hashes, and errors.
The public CSV contains cell summaries rather than individual request traces.

## Metric definitions

- **TTFT**: first observed non-empty token chunk time minus request start.
- **E2E**: final completion time minus request start.
- **TPOT**: time from first to final observed token divided by output tokens
  minus one. It is undefined for one-token outputs.
- **ITL**: interval between adjacent stream chunks only when both chunks
  contain exactly one token. This avoids pretending that merged chunks expose
  exact token arrival times.
- **Requests/s**: successful requests divided by cell wall time.
- **Output tokens/s**: successful completion tokens divided by cell wall time.

Percentiles use the nearest-rank definition.

## Validation

Run:

```bash
python analysis/validate_public_results.py
```

The validator reconstructs the 480-cell product, checks all 728 latency rows,
verifies 58,240 successful requests, checks the stored hash-pair validation,
and rejects known internal provenance columns.

The original collection process also checked output lengths, matrix coverage,
request errors, paired input hashes, and quality-probe group coverage before
the aggregate files were selected for publication.
