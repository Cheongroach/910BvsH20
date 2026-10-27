"""Run the same matrix workload through CUDA or Ascend PyTorch profilers."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path


def package_versions():
    versions = {}
    for package in ("torch", "torch-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def profiler_api(device, output_dir, steps):
    import torch

    if device == "cuda":
        return torch.profiler, [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], "self_cuda_time_total", {}

    import torch_npu

    return torch_npu.profiler, [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ], None, {
        "schedule": torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=steps,
            repeat=1,
        ),
        "on_trace_ready": torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir),
            analyse_flag=True,
        ),
    }


def synchronize(torch, device):
    getattr(torch, device).synchronize()


def matrix_step(torch, x, y):
    return torch.nn.functional.silu(x @ y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "npu"], required=True)
    parser.add_argument("--matrix-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", default="/tmp/profile-probe")
    parser.add_argument("--workload-only", action="store_true")
    args = parser.parse_args()

    import torch

    if args.device == "npu":
        import torch_npu  # noqa: F401

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{args.device}-torch-profiler.json"
    summary_path = output_dir / f"{args.device}-summary.json"

    torch.manual_seed(20261027)
    x = torch.randn(
        args.matrix_size,
        args.matrix_size,
        dtype=torch.bfloat16,
        device=args.device,
    )
    y = torch.randn_like(x)

    for _ in range(args.warmup):
        matrix_step(torch, x, y)
    synchronize(torch, args.device)

    if args.workload_only:
        wall_start = time.perf_counter()
        for _ in range(args.steps):
            matrix_step(torch, x, y)
        synchronize(torch, args.device)
        summary = {
            "device": args.device,
            "device_name": getattr(torch, args.device).get_device_name(0),
            "versions": package_versions(),
            "matrix_shape": [args.matrix_size, args.matrix_size],
            "dtype": "bfloat16",
            "warmup": args.warmup,
            "steps": args.steps,
            "wall_seconds": time.perf_counter() - wall_start,
            "mode": "workload_only",
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print("PROFILE_PROBE " + json.dumps(summary, sort_keys=True), flush=True)
        return

    profiler, activities, sort_key, extra_profiler_args = profiler_api(
        args.device,
        output_dir,
        args.steps,
    )
    wall_start = time.perf_counter()
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        **extra_profiler_args,
    ) as prof:
        for _ in range(args.steps):
            matrix_step(torch, x, y)
            if args.device == "npu":
                prof.step()
    synchronize(torch, args.device)
    wall_seconds = time.perf_counter() - wall_start

    table = None
    if args.device == "cuda":
        prof.export_chrome_trace(str(trace_path))
        table = prof.key_averages().table(sort_by=sort_key, row_limit=12)
        trace_files = [trace_path]
    else:
        trace_files = sorted(path for path in output_dir.rglob("*") if path.is_file())

    summary = {
        "device": args.device,
        "device_name": getattr(torch, args.device).get_device_name(0),
        "versions": package_versions(),
        "matrix_shape": [args.matrix_size, args.matrix_size],
        "dtype": "bfloat16",
        "warmup": args.warmup,
        "profiled_steps": args.steps,
        "wall_seconds": wall_seconds,
        "trace_files": [
            {"path": str(path), "bytes": path.stat().st_size}
            for path in trace_files
        ],
        "trace_bytes": sum(path.stat().st_size for path in trace_files),
        "sort_key": sort_key,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("PROFILE_PROBE " + json.dumps(summary, sort_keys=True), flush=True)
    if table:
        print(table, flush=True)


if __name__ == "__main__":
    main()
