#!/usr/bin/env python3
"""Collect OTM runs into a baseline-referenced comparison table."""

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "Image AUROC",
    "Image AUPR",
    "Pixel AUROC",
    "AU-PRO",
    "Image F1",
    "Pixel F1",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def collect(root: Path):
    rows = []
    for result_path in sorted(root.rglob("benchmark_results.csv")):
        config_path = result_path.with_name("config.json")
        if not config_path.exists():
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = pd.read_csv(result_path)

        if config.get("k_shot") is not None:
            shot = str(config["k_shot"])
        elif config.get("batched_zero_shot", False):
            shot = "zero"
        else:
            shot = "full"

        for _, item in result.iterrows():
            row = {
                "Dataset": config.get("dataset_name"),
                "K Shot": shot,
                "Variant": config.get("otm_variant", "baseline"),
                "Category": item["Category"],
                "Seed": item["Seed"],
                "Bank Size": config.get("otm_bank_size"),
            }
            for metric in METRICS:
                row[metric] = item.get(metric)
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No benchmark_results.csv/config.json pairs found below {root}."
        )

    table = pd.DataFrame(rows)
    order = {"baseline": 0, "tangent": 1, "raw_knn": 2, "otm_full": 3}
    table["_order"] = table["Variant"].map(order).fillna(99)
    table = table.sort_values(
        ["Dataset", "K Shot", "Category", "_order", "Seed"]
    ).drop(columns="_order")

    keys = ["Dataset", "K Shot", "Category", "Seed"]
    baseline_rows = table[table["Variant"] == "baseline"]
    for metric in ["Image AUROC", "Image AUPR", "Pixel AUROC", "AU-PRO"]:
        baseline = baseline_rows.set_index(keys)[metric].to_dict()
        table[f"Delta {metric}"] = [
            value - baseline.get((dataset, shot, category, seed), float("nan"))
            for value, dataset, shot, category, seed in zip(
                table[metric],
                table["Dataset"],
                table["K Shot"],
                table["Category"],
                table["Seed"],
            )
        ]
    return table.reset_index(drop=True)


def main():
    args = parse_args()
    root = args.root.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else root / "otm_ablation_summary.csv"
    )
    table = collect(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, float_format="%.6f")

    display_columns = [
        "Dataset",
        "K Shot",
        "Variant",
        "Category",
        "Image AUROC",
        "Image AUPR",
        "Pixel AUROC",
        "AU-PRO",
        "Delta Pixel AUROC",
        "Delta AU-PRO",
    ]
    print(table[display_columns].to_string(index=False, float_format="%.4f"))
    print(f"\nSummary saved to: {output}")


if __name__ == "__main__":
    main()
