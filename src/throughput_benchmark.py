"""Real-text fixed-token inference and independent task-quality probes."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import string
import time
from pathlib import Path


def emit(event, **fields):
    print("IB_RESULT " + json.dumps(dict(event=event, **fields), ensure_ascii=False), flush=True)


def percentile(values, q):
    return sorted(values)[min(len(values)-1, math.ceil(q*len(values))-1)] if values else None


def normalize(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="4b")
    p.add_argument("--config", default="baseline")
    p.add_argument("--device", choices=["cuda", "npu"], required=True)
    p.add_argument("--root", default="/tmp/ib")
    p.add_argument("--suite", choices=["sweep", "ablation", "quality", "confirm"], default="sweep")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--budget", type=int, default=4096)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--lengths", default="1024,2048")
    p.add_argument("--outputs", default="1,64,128")
    p.add_argument("--max-seqs",type=int,default=128)
    args = p.parse_args()
    import torch
    if args.device == "npu":
        import torch_npu
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    root = Path(args.root)
    manifest = json.loads((root / "manifest.json").read_text())
    model = next(m for m in manifest["models"] if m["tag"] == args.model)
    rows = [json.loads(line) for line in (root / "workload.jsonl").read_text().splitlines()]
    path = str(root / args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    def chat_ids(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return tokenizer.encode(text, add_special_tokens=False)
    config = args.config
    graph = config in ["graph", "combo", "fp8", "mtp", "graphonly", "compileonly", "fp8mtp", "fp8graph"]
    chunk = config in ["chunk", "combo", "fp8", "mtp", "prefix", "prefixoff", "fp8mtp"]
    prefix = config == "prefix"
    kw = dict(model=path, dtype="bfloat16", tensor_parallel_size=1,
              enforce_eager=not graph, enable_prefix_caching=prefix,
              enable_chunked_prefill=chunk, max_model_len=2560,
              max_num_seqs=args.max_seqs, max_num_batched_tokens=args.budget if chunk else 32768,
              gpu_memory_utilization=0.88, seed=20261019, disable_log_stats=False)
    if args.device == "npu":
        kw["block_size"] = 128  # Ascend Attention backend requires a supported KV block size.
    if args.model in ["2b","4b","9b"]:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if config in ["fp8", "fp8mtp", "fp8graph"]:
        kw["quantization"] = "fp8"
    if config in ["mtp", "fp8mtp"]:
        kw["speculative_config"] = {"method": "qwen3_5_mtp", "num_speculative_tokens": 2}
    if config == "graphonly":
        kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL"}
    if config == "compileonly":
        kw["compilation_config"] = {"mode": 3, "cudagraph_mode": "NONE"}
    versions = {}
    for pkg in ["torch", "torch-npu", "vllm", "vllm-ascend", "transformers"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass
    emit("engine_start", model=args.model, config=config, kwargs=kw, versions=versions,
         cpu_threads=os.environ.get("OMP_NUM_THREADS"), model_revision=model["hf_revision"], workload_sha256=manifest["dataset"]["selected_sha256"])
    start = time.perf_counter()
    llm = LLM(**kw)
    emit("engine_ready", model=args.model, config=config, load_seconds=time.perf_counter()-start,
         device_name=getattr(torch, args.device).get_device_name(0))

    def prompt_tokens(i, length, task="generate", shared=False):
        row = rows[i % len(rows)]
        task_text = {
            "score": "Does the candidate answer correctly answer the question? Output only 0 or 1.",
            "extract": "Extract the shortest answer from the context. Output only the answer.",
            "generate": "Explain the answer using the provided context in a concise paragraph.",
        }[task]
        messages = [{"role": "user", "content":
                     f"Request {i:08d}\n{task_text}\nContext:\n{row['context']}\n"
                     f"Question: {row['question']}\nCandidate answer: {row['answers'][0]}"}]
        suffix = chat_ids(messages)
        # Prepend other real contexts as reference material, leaving the complete QA suffix.
        reference = []
        j = 0
        while len(reference) + len(suffix) < length:
            reference.extend(tokenizer.encode(rows[(i+j+7) % len(rows)]["context"],
                                              add_special_tokens=False))
            j += 1
        result = (reference[:max(0, length-len(suffix))] + suffix)[-length:]
        if shared:
            # Shared reference prefix is deliberate, paired in both cache configurations.
            common = tokenizer.encode("\n".join(r["context"] for r in rows[:16]),
                                      add_special_tokens=False)
            result = common[:length-128] + result[-128:]
        else:
            # Distinct first block prevents accidentally benchmarking shared reference caching.
            unique = tokenizer.encode(f"Document identifier {i:08d}. ", add_special_tokens=False)
            result[:len(unique)] = unique
        assert len(result) == length
        return {"prompt_token_ids": result}

    def measure(length, output_len, batch, repeat, cache="none"):
        count = max(8, min(256, batch*2))
        task = {1: "score", 64: "extract", 128: "generate"}[output_len]
        # Only prefix tests intentionally reuse exact requests between repetitions.
        offset = 0 if cache != "none" else 128+repeat*128
        prompts = [prompt_tokens(offset+i, length, task, cache != "none") for i in range(count)]
        params = SamplingParams(temperature=0, max_tokens=output_len, min_tokens=output_len,
                                ignore_eos=True)
        if cache == "cold" and hasattr(llm, "reset_prefix_cache"):
            llm.reset_prefix_cache()
        warm = [prompt_tokens(i+10000, length, task) for i in range(min(batch, 8))]
        t = time.perf_counter()
        llm.generate(warm, SamplingParams(temperature=0, max_tokens=min(output_len, 8),
                                         ignore_eos=True), use_tqdm=False)
        warm_seconds = time.perf_counter()-t
        if cache == "warm":
            llm.generate(prompts, params, use_tqdm=False)
        input_hash = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        completed = []
        group_latencies = []
        ttfts = []
        request_latencies = []
        t = time.perf_counter()
        for begin in range(0, count, batch):
            a = time.perf_counter()
            out = llm.generate(prompts[begin:begin+batch], params, use_tqdm=False)
            group_latencies.append(time.perf_counter()-a)
            completed.extend(out)
            for r in out:
                metrics = getattr(r, "metrics", None)
                if metrics and getattr(metrics, "first_token_time", None) is not None:
                    ttfts.append(metrics.first_token_time-metrics.arrival_time)
                if metrics and getattr(metrics, "finished_time", None) is not None:
                    request_latencies.append(metrics.finished_time-metrics.arrival_time)
        seconds = time.perf_counter()-t
        assert len(completed) == count
        sizes = [len(r.outputs[0].token_ids) for r in completed]
        assert all(n == output_len for n in sizes), sizes
        output_hash = hashlib.sha256(json.dumps(
            [r.outputs[0].token_ids for r in completed]).encode()).hexdigest()
        mem = getattr(torch, args.device)
        emit("measurement", model=args.model, config=config, suite=args.suite,
             input_tokens=length, output_tokens=output_len, batch=batch, repeat=repeat,
             cache=cache, requests=count, seconds=seconds,
             requests_s=count/seconds, input_tokens_s=count*length/seconds,
             output_tokens_s=sum(sizes)/seconds, output_sizes=sizes,
             group_latency_p50_s=statistics.median(group_latencies),
             request_latency_p50_s=statistics.median(request_latencies) if request_latencies else None,
             ttft_p50_s=statistics.median(ttfts) if ttfts else None,
             ttft_p95_s=percentile(ttfts, .95), warmup_seconds=warm_seconds,
             allocated_gib=None, reserved_gib=None,
             memory_note="Engine process owns device memory; consult engine logs",
             input_sha256=input_hash, output_sha256=output_hash,
             max_num_seqs=kw["max_num_seqs"], block_size=kw.get("block_size"),
             budget=kw["max_num_batched_tokens"])

    if args.suite == "quality":
        quality_prompts, expected = [], []
        for i, row in enumerate(rows[:64]):
            answer = row["answers"][0] if i%2 == 0 else rows[(i+19)%len(rows)]["answers"][0]
            correct = normalize(answer) in [normalize(a) for a in row["answers"]]
            for task in ["score", "extract"]:
                instruction = ("Output only 1 if the candidate correctly answers the question, "
                               "otherwise output only 0.") if task == "score" else (
                                   "Extract the shortest answer to the question. Output only the answer.")
                messages = [{"role": "user", "content":
                             f"{instruction}\nContext: {row['context']}\n"
                             f"Question: {row['question']}" + (f"\nCandidate: {answer}" if task=="score" else "")}]
                ids = chat_ids(messages)
                quality_prompts.append({"prompt_token_ids": ids[-2048:]})
                expected.append((task, row, correct))
        out = llm.generate(quality_prompts, SamplingParams(temperature=0, max_tokens=64),
                           use_tqdm=False)
        for r, (task, row, correct) in zip(out, expected):
            text = r.outputs[0].text.strip()
            hit = (text == str(int(correct))) if task == "score" else (
                normalize(text) in [normalize(a) for a in row["answers"]])
            emit("quality", model=args.model, config=config, task=task, id=row["id"],
                 expected=str(int(correct)) if task=="score" else row["answers"],
                 output=text, correct=hit, quality_protocol="candidate_only_in_score_v2", output_tokens=len(r.outputs[0].token_ids),
                 input_tokens=len(r.prompt_token_ids), finish_reason=r.outputs[0].finish_reason)
    else:
        batches = [1, 8, 32, 128] if args.suite == "sweep" else [args.batch]
        repeats = 1 if args.suite == "sweep" else args.repeats
        for length in map(int,args.lengths.split(",")):
            for output_len in map(int,args.outputs.split(",")):
                for batch in batches:
                    for repeat in range(repeats):
                        measure(length, output_len, batch, repeat)
        if config in ["prefixoff", "prefix"]:
            for length in map(int,args.lengths.split(",")):
                for cache in ["cold", "warm"]:
                    for repeat in range(args.repeats):
                        measure(length, 1, 32, repeat, cache)
    emit("complete", model=args.model, config=config, suite=args.suite)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        emit("failed", error_type=type(error).__name__, error=str(error))
        raise
