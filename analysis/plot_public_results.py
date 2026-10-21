"""Rebuild the primary public comparison chart from aggregate CSV data."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "latency" / "summary.csv"
OUTPUT = ROOT / "results" / "latency" / "charts" / "generation-throughput.png"
MODELS = ["2b", "4b", "9b", "14b", "35-27b", "36-27b", "38-27b"]
LABELS = ["Qwen3.5 2B", "Qwen3.5 4B", "Qwen3.5 9B", "Qwen3 14B",
          "Qwen3.5 27B", "Qwen3.6 27B", "Qwen3.8 27B"]


def selected_rows():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row
        for row in rows
        if row["config"] == "combo"
        and row["mode"] == "online"
        and row["arrival"] == "closed_loop"
        and row["concurrency"] == "32"
        and row["input_tokens"] == "1024"
        and row["output_tokens"] == "128"
    ]
    values = {
        (row["model"], row["hardware"]): float(row["output_tokens_s"])
        for row in rows
    }
    expected = {(model, hardware) for model in MODELS for hardware in ("h20", "npu")}
    if set(values) != expected:
        raise RuntimeError(f"chart selection mismatch: found {len(values)}/{len(expected)} rows")
    return values


def main():
    values = selected_rows()
    h20 = [values[(model, "h20")] for model in MODELS]
    npu = [values[(model, "npu")] for model in MODELS]

    x = np.arange(len(MODELS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12.8, 6.8), layout="constrained")
    fig.patch.set_facecolor("#f7f7f5")
    ax.set_facecolor("#f7f7f5")

    bars_h20 = ax.bar(x - width / 2, h20, width, label="NVIDIA H20", color="#2678b8")
    bars_npu = ax.bar(x + width / 2, npu, width, label="Ascend 910B4", color="#d05a47")

    ax.set_title("Generation throughput by model", fontsize=17, pad=15)
    ax.text(
        0,
        1.01,
        "Online closed-loop, concurrency 32, 1,024 input tokens, 128 output tokens",
        transform=ax.transAxes,
        color="#555555",
        fontsize=10.5,
    )
    ax.set_ylabel("Output tokens / second")
    ax.set_xticks(x, LABELS)
    ax.grid(axis="y", color="#d7d7d2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    ceiling = max(max(h20), max(npu))
    for bars in (bars_h20, bars_npu):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ceiling * 0.012,
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
            )
    ax.set_ylim(0, ceiling * 1.17)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
