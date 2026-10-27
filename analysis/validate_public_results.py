"""Validate the public aggregate benchmark datasets."""

import csv
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT_CSV = ROOT / "results" / "throughput" / "summary.csv"
THROUGHPUT_VALIDATION = ROOT / "results" / "throughput" / "validation.json"
LATENCY_CSV = ROOT / "results" / "latency" / "summary.csv"
LATENCY_VALIDATION = ROOT / "results" / "latency" / "validation.json"
OBSERVABILITY_SUMMARY = ROOT / "results" / "observability" / "monitoring-summary.json"
PROFILING_SUMMARY = ROOT / "results" / "observability" / "profiling-summary.json"

FORBIDDEN_COLUMNS = {
    "source",
    "source_file",
    "job_id",
    "job_url",
    "log_path",
    "worker_path",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_throughput():
    fields, rows = read_csv(THROUGHPUT_CSV)
    metadata = json.loads(THROUGHPUT_VALIDATION.read_text(encoding="utf-8"))

    require(not (FORBIDDEN_COLUMNS & set(fields)), "throughput CSV exposes internal columns")
    require(sum(int(row["n"]) for row in rows) == 1326, "unexpected measurement count")
    require(metadata["measurements"] == 1326, "validation measurement count changed")
    require(metadata["errors"] == [], "throughput validation contains errors")
    require(metadata["input_hash_mismatches"] == [], "throughput input hashes differ")
    require(metadata["cross_hardware_repeat_pairs"] == 540, "repeat pairing changed")
    require(metadata["quality_answers"] == 2176, "quality answer count changed")

    dimensions = itertools.product(
        ("h20", "npu"),
        ("2b", "4b", "9b", "14b", "q3-4b"),
        ("baseline", "combo"),
        ("1024", "2048"),
        ("1", "64", "128"),
        ("1", "8", "32", "128"),
    )
    expected = set(dimensions)
    sweep_rows = [row for row in rows if row["suite"] == "sweep"]
    actual = {
        (
            row["hardware"],
            row["model"],
            row["config"],
            row["input_tokens"],
            row["output_tokens"],
            row["batch"],
        )
        for row in sweep_rows
    }
    require(actual == expected, f"throughput matrix mismatch: {len(actual)}/480 cells")
    require(metadata["required_sweep_cells"] == 480, "validation cell count changed")
    require(metadata["missing_sweep_cells"] == [], "throughput cells are missing")

    return {
        "aggregate_rows": len(rows),
        "measurements": sum(int(row["n"]) for row in rows),
        "sweep_cells": len(actual),
        "quality_answers": metadata["quality_answers"],
    }


def validate_latency():
    fields, rows = read_csv(LATENCY_CSV)
    metadata = json.loads(LATENCY_VALIDATION.read_text(encoding="utf-8"))

    require(not (FORBIDDEN_COLUMNS & set(fields)), "latency CSV exposes internal columns")
    require(len(rows) == 728, f"unexpected latency cell count: {len(rows)}")
    require(sum(int(row["requests"]) for row in rows) == 58240, "request count changed")
    require(
        sum(int(row["successful"]) for row in rows) == 58240,
        "latency dataset contains unsuccessful requests",
    )
    require({row["hardware"] for row in rows} == {"h20", "npu"}, "hardware set changed")
    require(all(row["comparison_eligible"] == "True" for row in rows), "ineligible rows found")

    require(metadata["selected_cells"] == metadata["expected_cells"] == 728, "cell plan changed")
    require(
        metadata["selected_requests"] == metadata["expected_requests"] == 58240,
        "request plan changed",
    )
    require(metadata["request_errors"] == 0, "request errors are present")
    require(metadata["missing_cells"] == 0, "latency cells are missing")
    require(metadata["unexpected_cells"] == 0, "unexpected latency cells are present")
    require(metadata["wrong_cell_request_counts"] == 0, "per-cell request counts differ")
    require(metadata["cross_hardware_input_pairs"] == 23520, "pair count changed")
    require(metadata["unpaired_closed_loop_requests"] == 0, "unpaired requests are present")
    require(metadata["input_hash_mismatches"] == 0, "cross-hardware inputs differ")

    return {
        "cells": len(rows),
        "successful_requests": sum(int(row["successful"]) for row in rows),
        "cross_hardware_input_pairs": metadata["cross_hardware_input_pairs"],
    }


def validate_observability():
    summary = json.loads(OBSERVABILITY_SUMMARY.read_text(encoding="utf-8"))
    h20 = summary["h20"]
    npu = summary["ascend_910b4"]

    require(h20["samples"] == 41, "H20 monitoring sample count changed")
    require(npu["samples"] == 54, "910B4 monitoring sample count changed")
    require(
        h20["metrics"]["duty_cycle_percent"]["metric"]
        == "lab.arnold.instance_monitor.gpu_duty_cycle",
        "H20 duty-cycle metric changed",
    )
    require(
        npu["metrics"]["memory_bandwidth_utilization_percent"]["metric"]
        == "lab.arnold.instance_monitor.npu_hbm_bandutil",
        "910B4 bandwidth metric changed",
    )
    require(
        h20["metrics"]["device_memory_total_bytes"]["max"] == 102625181696,
        "H20 device-memory total changed",
    )
    require(
        npu["metrics"]["hbm_total_panel_native"]["max"] == 65536,
        "910B4 panel-native HBM total changed",
    )
    require(
        npu["metrics"]["hbm_used"]["max"] is None,
        "missing 910B4 used-memory series was converted to a value",
    )
    require(
        npu["metrics"]["cpu_panel"]["max"] is None,
        "empty 910B4 CPU panel was converted to a value",
    )

    return {
        "source": summary["source"],
        "h20_samples": h20["samples"],
        "npu_samples": npu["samples"],
        "missing_series_preserved": 2,
    }


def validate_profiling():
    summary = json.loads(PROFILING_SUMMARY.read_text(encoding="utf-8"))
    h20 = summary["h20"]
    npu = summary["ascend_910b4"]

    require(h20["job_status"] == "done", "H20 profiler job is not done")
    require(npu["job_status"] == "done", "910B4 profiler job is not done")
    require(h20["trial_status"] == "finished", "H20 profiler trial did not finish")
    require(npu["trial_status"] == "finished", "910B4 profiler trial did not finish")
    require(h20["pod_exit_status"] == 0, "H20 profiler pod did not exit cleanly")
    require(npu["pod_exit_status"] == 0, "910B4 profiler pod did not exit cleanly")
    require(h20["torch_profiler"]["return_code"] == 0, "H20 torch profiler failed")
    require(npu["torch_npu_profiler"]["return_code"] == 0, "torch-npu profiler failed")
    require(npu["msprof"]["return_code"] == 0, "msprof failed")
    require(h20["torch_profiler"]["trace_bytes"] == 82151, "H20 trace size changed")
    require(
        npu["torch_npu_profiler"]["representative_outputs"]["trace_view.json"] == 158906,
        "910B4 trace-view size changed",
    )
    require(
        not h20["tool_inventory"]["nsys"] and not h20["tool_inventory"]["ncu"],
        "H20 image tool inventory changed",
    )
    require(
        npu["environment"]["cann_path_observed"].endswith("cann-8.5.1"),
        "910B4 probe CANN evidence changed",
    )

    return {
        "h20_torch_profiler": "passed",
        "npu_torch_profiler": "passed",
        "npu_msprof": "passed",
        "persisted_assets": 0,
    }


def main():
    result = {
        "throughput": validate_throughput(),
        "latency": validate_latency(),
        "observability": validate_observability(),
        "profiling": validate_profiling(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Public result validation passed.")


if __name__ == "__main__":
    main()
