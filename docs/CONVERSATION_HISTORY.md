# Sanitized project conversation history

This is a curated project history derived from the user-visible requests and
the resulting engineering work. It is not a verbatim transcript. Internal
paths, infrastructure identifiers, credentials, raw logs, and hidden
instructions are intentionally excluded.

## 1. Initial comparison

The project began with a request to compare H20 and Ascend 910B4 inference for
Qwen models from 2B to 14B under roughly 1K-to-2K token prompts. The desired
workload included judgment, answer extraction, and longer-form generation
rather than synthetic empty prompts.

The scope evolved into a controlled matrix of model size, input length, output
length, batch size, and hardware.

## 2. Baselines and optimization ablations

The next requirement was to establish a common baseline and test optimizations
independently. The benchmark therefore separated eager baseline runs from
graph/compile behavior, chunked prefill, prefix caching, FP8, MTP, and a
combined configuration.

This phase also introduced exact prompt lengths, deterministic output lengths,
warmups, unique prefixes outside intentional cache tests, and artifact hashes.

## 3. Request-level latency

The model set expanded to include same-size Qwen3.5, Qwen3.6, and Qwen3.8-27B
comparisons. The requested metrics expanded beyond aggregate throughput to
TTFT, TPOT, ITL, E2E P95/P99, and both offline and OpenAI-compatible streaming
paths.

The implementation added isolated loopback endpoints, closed-loop concurrency,
open-loop Poisson arrivals, SSE parsing, exact token-chunk accounting, and
cross-hardware input-hash pairing.

## 4. Completing and checking the matrix

The project was then driven to full planned coverage rather than reporting a
partial sample. The completed public selection contains:

- 480/480 unique throughput sweep cells
- 1,326 valid underlying throughput measurements
- 728/728 latency cells
- 58,240 successful latency requests
- 23,520 paired cross-hardware request inputs with no hash mismatch

The 27B high-concurrency regression was reproduced across three same-size model
revisions. The conclusion was deliberately limited to the observed deployment
behavior because no profiler evidence established a chip-level cause.

## 5. Interpreting the work for a resume

The user asked how to read the result and why it matters for a resume. The
answer emphasized the engineering evidence:

- experiment design and confounder control
- inference metric semantics
- asynchronous and streaming request instrumentation
- backend-aware optimization testing
- data validation and restrained root-cause claims

The work was then rewritten using the STAR format. “480 cells” was clarified
as the number of unique configurations in the Cartesian experiment matrix,
not the number of requests or scheduler jobs.

## 6. Naming and publication

The project was named **HeteroInferLab** to reflect controlled experiments
across heterogeneous inference stacks.

The final request was to publish the code, conclusions, and relevant
conversation history to `Cheongroach/910BvsH20`. The public repository contains
portable benchmark code, pinned manifests, a selected workload, aggregate
results, charts, validation scripts, methodology, conclusions, this sanitized
history, and a resume narrative.

Raw request traces and scheduler logs are not published because they contain
environment-specific metadata and are unnecessarily large. Hidden prompts and
private infrastructure details are not part of the project record.
