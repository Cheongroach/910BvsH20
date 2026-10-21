"""Measure async engine token delivery and OpenAI SSE client latency."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from pathlib import Path


def emit(event, **fields):
    print("LAT_RESULT " + json.dumps(dict(event=event, **fields)), flush=True)


def quantile(values, q):
    return sorted(values)[min(len(values)-1, math.ceil(q*len(values))-1)] if values else None


def summarize(records, elapsed, **shape):
    ok = [r for r in records if not r.get("error")]
    fields = ["ttft_s", "e2e_s", "tpot_s", "itl_s"]
    summary = dict(requests=len(records), successful=len(ok), seconds=elapsed,
                   requests_s=len(ok)/elapsed,
                   output_tokens_s=sum(r["tokens"] for r in ok)/elapsed)
    for key in fields:
        values = ([x for r in ok for x in r[key]] if key == "itl_s"
                  else [r[key] for r in ok if r[key] is not None])
        for q in [.5, .95, .99]:
            summary[f"{key}_p{int(q*100)}"] = quantile(values, q)
    summary["merged_token_chunks"] = sum(r.get("merged_token_chunks", 0) for r in ok)
    emit("summary", **shape, **summary)


def make_prompts(root, model, length, output, count, offset):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(root/model), local_files_only=True)
    rows = [json.loads(l) for l in (root/"workload.jsonl").read_text().splitlines()]
    prompts = []
    for i in range(count):
        row = rows[(i+offset) % len(rows)]
        task = {1:"Output only 0 or 1: does the candidate answer the question?",
                64:"Extract the shortest answer. Output only the answer.",
                128:"Explain the answer in a concise paragraph."}[output]
        content = f"{task}\nContext: {row['context']}\nQuestion: {row['question']}"
        if output == 1:
            content += f"\nCandidate: {row['answers'][0]}"
        ids = tok.encode(tok.apply_chat_template([{"role":"user","content":content}],
                         tokenize=False, add_generation_prompt=True, enable_thinking=False),
                         add_special_tokens=False)
        reference = tok.encode("\n".join(r["context"] for r in rows[:16]),
                               add_special_tokens=False)
        ids = (reference * (length//max(1,len(reference))+2))[:max(0,length-len(ids))] + ids
        ids = ids[-length:]
        unique = tok.encode(f"Document {offset+i:08d}: ", add_special_tokens=False)
        ids[:len(unique)] = unique
        assert len(ids) == length
        prompts.append(ids)
    return prompts


async def offline(args):
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    kw = engine_kwargs(args)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    emit("engine_ready", mode="offline", kwargs=kw)
    async def request(ids, output, rid):
        start = time.perf_counter()
        stamps, chunks, token_ids = [], [], []
        params = SamplingParams(temperature=0, min_tokens=output, max_tokens=output,
                                ignore_eos=True, output_kind=RequestOutputKind.DELTA)
        async for response in engine.generate({"prompt_token_ids":ids}, params, rid):
            now = time.perf_counter()
            delta = list(response.outputs[0].token_ids)
            if delta:
                stamps.append(now-start)
                chunks.append(len(delta))
                token_ids.extend(delta)
        end = time.perf_counter()-start
        assert len(token_ids) == output, (rid, len(token_ids), output)
        # Only adjacent chunks each containing one token yield an exact ITL.
        itls = [stamps[i]-stamps[i-1] for i in range(1,len(stamps))
                if chunks[i] == chunks[i-1] == 1]
        return dict(id=rid, ttft_s=stamps[0], e2e_s=end,
                    tpot_s=(stamps[-1]-stamps[0])/(output-1) if output>1 else None,
                    itl_s=itls, token_arrival_s=stamps, chunk_tokens=chunks,
                    merged_token_chunks=sum(n>1 for n in chunks), tokens=output,
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    output_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest())
    try:
        for length in [1024,2048]:
            for output in [1,64,128]:
                for concurrency in [1,32]:
                    count = max(16,concurrency*2)
                    prompts = make_prompts(args.root,args.model,length,output,count,128)
                    # Full shape/full output warmup prevents first decode compile contamination.
                    await asyncio.gather(*(request(ids,output,f"warm-{i}") for i,ids in enumerate(prompts[:concurrency])))
                    sem = asyncio.Semaphore(concurrency)
                    async def run(i):
                        async with sem:
                            r = await request(prompts[i],output,f"off-{length}-{output}-{concurrency}-{i}")
                            r.update(mode="offline", model=args.model, config=args.config,
                                     input_tokens=length, output_tokens=output, concurrency=concurrency)
                            return r
                    start = time.perf_counter()
                    records = await asyncio.gather(*(run(i) for i in range(count)))
                    elapsed = time.perf_counter()-start
                    for record in records:
                        emit("request", **record)
                    summarize(records,elapsed,mode="offline",model=args.model,
                              config=args.config,input_tokens=length,output_tokens=output,
                              concurrency=concurrency,arrival="closed_loop")
    finally:
        engine.shutdown()


def engine_kwargs(args):
    graph = args.config in ["graph","combo","fp8"]
    kw = dict(model=str(args.root/args.model), dtype="bfloat16",
              enforce_eager=not graph, enable_prefix_caching=False,
              enable_chunked_prefill=args.config in ["combo","fp8"],
              max_num_batched_tokens=4096, max_model_len=2560, max_num_seqs=32,
              gpu_memory_utilization=.92 if "27b" in args.model else .88,
              tensor_parallel_size=1, seed=20261019, disable_log_stats=False,
              limit_mm_per_prompt={"image":0,"video":0})
    if args.device == "npu":
        kw["block_size"] = 128
    if args.config == "fp8":
        kw["quantization"] = "fp8"
    return kw


async def online(args):
    # A fixed-port controller cannot prove which parallel server owns the endpoint.
    if args.url == "http://127.0.0.1:18080":
        raise RuntimeError("Unisolated HTTP endpoint: use the dynamic-port controller")
    import aiohttp
    import random
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0,force_close=True), trust_env=False, timeout=aiohttp.ClientTimeout(total=300)) as session:
        async def request(ids, output, rid, scheduled=None):
            if scheduled is not None:
                await asyncio.sleep(max(0,scheduled-time.perf_counter()))
            start = time.perf_counter()
            stamps, deltas, usage = [], [], None
            body = dict(model=args.model, prompt=ids, temperature=0,max_tokens=output,
                        min_tokens=output,ignore_eos=True,stream=True,
                        stream_options={"include_usage":True},
                        return_token_ids=True, seed=20261019)
            try:
                async with session.post(args.url+"/v1/completions",json=body) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:400]}")
                    # StreamReader line iteration handles TCP fragmentation and coalescing.
                    async for raw in response.content:
                        if not raw.startswith(b"data: "):
                            continue
                        payload = raw[6:].strip()
                        if payload == b"[DONE]":
                            break
                        obj = json.loads(payload)
                        if obj.get("usage"):
                            usage = obj["usage"]
                        for choice in obj.get("choices",[]):
                            tids = choice.get("token_ids")
                            text = choice.get("text","")
                            if tids if tids is not None else text:
                                stamps.append(time.perf_counter()-start)
                                deltas.append(len(tids) if tids is not None else None)
                end = time.perf_counter()-start
                if not stamps or usage is None or usage["completion_tokens"] != output or usage["prompt_tokens"] != len(ids):
                    raise RuntimeError(f"stream contract: chunks={len(stamps)}, usage={usage}")
                exact = all(n is not None for n in deltas)
                itls = [stamps[i]-stamps[i-1] for i in range(1,len(stamps))
                        if exact and deltas[i] == deltas[i-1] == 1]
                return dict(id=rid,ttft_s=stamps[0],e2e_s=end,
                            tpot_s=(stamps[-1]-stamps[0])/(output-1) if output>1 else None,
                            itl_s=itls,token_arrival_s=stamps,chunk_tokens=deltas,
                            merged_token_chunks=sum(n>1 for n in deltas if n is not None),
                            token_ids_available=exact,tokens=output,
                            schedule_lag_s=start-scheduled if scheduled is not None else None,
                            input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            except Exception as e:
                return dict(id=rid,error=str(e),ttft_s=None,e2e_s=time.perf_counter()-start,
                            tpot_s=None,itl_s=[],tokens=0)
        for length in [1024,2048]:
            for output in [1,64,128]:
                planned = [r for r in args.resume_cells if r["config"]==args.config
                           and r["input_tokens"]==length and r["output_tokens"]==output]
                if args.resume_cells and not planned:
                    continue
                prompts = make_prompts(args.root,args.model,length,output,200,512)
                for concurrency in [8,32]:
                    closed = not args.resume_cells or any(
                        r["arrival"]=="closed_loop" and r["concurrency"]==concurrency for r in planned)
                    poisson = [r for r in planned if r["arrival"]=="poisson"] if concurrency==32 else []
                    if not closed and not poisson:
                        continue
                    warm = await asyncio.gather(*(request(prompts[i],output,f"warm-{i}") for i in range(concurrency)))
                    if any(r.get("error") for r in warm):
                        raise RuntimeError(warm[0])
                    sem = asyncio.Semaphore(concurrency)
                    async def run(i):
                        async with sem:
                            r = await request(prompts[i],output,f"http-{length}-{output}-{concurrency}-{i}")
                            r.update(mode="online",model=args.model,config=args.config,
                                     input_tokens=length,output_tokens=output,concurrency=concurrency)
                            return r
                    capacity = None
                    if closed:
                        start = time.perf_counter()
                        records = await asyncio.gather(*(run(i) for i in range(100)))
                        elapsed = time.perf_counter()-start
                        for record in records:
                            emit("request", **record)
                        summarize(records,elapsed,mode="online",model=args.model,config=args.config,
                                  input_tokens=length,output_tokens=output,concurrency=concurrency,
                                  arrival="closed_loop")
                        capacity = len([r for r in records if not r.get("error")])/elapsed
                    if concurrency == 32 and output == 128 and args.config == "combo":
                        # Open-loop Poisson arrivals; all enqueue independently, without a client semaphore.
                        for load in [.5,.9]:
                            if args.resume_cells:
                                cell = next((r for r in poisson if r["load_fraction"]==load), None)
                                if cell is None:
                                    continue
                                # Keep the original measured closed-loop capacity when
                                # resuming, so a later run cannot silently change offered load.
                                rate = cell["offered_rate"]
                                if rate is None:
                                    assert capacity is not None, "Missing closed-loop capacity for resumed Poisson cell"
                                    rate = capacity*load
                            else:
                                rate = capacity*load
                            rng = random.Random(20261019)
                            origin = time.perf_counter()+.1
                            at = origin
                            scheduled = []
                            for _ in range(200):
                                at += rng.expovariate(rate)
                                scheduled.append(at)
                            async def offered(i):
                                r = await request(prompts[i],output,f"poisson-{load}-{i}",scheduled[i])
                                r.update(mode="online",model=args.model,config=args.config,
                                         input_tokens=length,output_tokens=output,arrival="poisson",
                                         offered_rate=rate,load_fraction=load)
                                return r
                            records = await asyncio.gather(*(offered(i) for i in range(200)))
                            elapsed = time.perf_counter()-origin
                            for record in records:
                                emit("request", **record)
                            summarize(records,elapsed,mode="online",
                                      model=args.model,config=args.config,input_tokens=length,
                                      output_tokens=output,concurrency=None,arrival="poisson",
                                      offered_rate=rate,load_fraction=load)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--config",required=True)
    p.add_argument("--device",required=True)
    p.add_argument("--mode",choices=["offline","online"],required=True)
    p.add_argument("--root",type=Path,default=Path("/tmp/ib"))
    p.add_argument("--url",default="http://127.0.0.1:18080")
    p.add_argument("--cells-file",type=Path)
    args = p.parse_args()
    args.resume_cells = json.loads(args.cells_file.read_text()) if args.cells_file else []
    versions = {}
    for name in ["torch","torch-npu","vllm","vllm-ascend","transformers"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    emit("start", model=args.model,config=args.config,mode=args.mode,versions=versions,
         protocol=("token_delivery_v4_unique_port"
                   if args.mode == "online" and args.url != "http://127.0.0.1:18080"
                   else "token_delivery_v3_fresh_http_connections"),
         http_connector_limit=0, url=args.url, resume_cells=args.resume_cells)
    asyncio.run(offline(args) if args.mode == "offline" else online(args))
    emit("complete",model=args.model,config=args.config,mode=args.mode)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit("failed",error_type=type(e).__name__,error=str(e))
        raise
