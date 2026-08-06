#!/usr/bin/env python3
"""Visualize one test image against every normal image as an independent 1-shot.

The DINOv2 backbone is loaded once and the test-image tokens are extracted
once.  Each image under ``train/good`` is then treated as the complete support
set of an independent SubspaceAD/OTM/COTA run.  Results are named with the
original support-image stem, e.g. ``000_heatmap.png``.

This file only orchestrates functions from ``tools/visualize_1shot.py`` and the
existing project modules.  It does not modify the benchmark implementation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import visualize_1shot as single


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
SUMMARY_FIELDS = [
    "support_id",
    "support_path",
    "status",
    "elapsed_seconds",
    "variant",
    "image_score_top_1_percent",
    "pca_retained_components",
    "feature_dimension",
    "effective_augmentation_count",
    "otm_bank_size",
    "raw_map_min",
    "raw_map_max",
    "heatmap_file",
    "overlay_file",
    "raw_map_file",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treat every image in train/good as an independent one-shot "
            "support and visualize one fixed test image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument(
        "--good_dir",
        type=Path,
        required=True,
        help="Directory containing the candidate normal support images.",
    )
    io_group.add_argument(
        "--anomaly_path",
        type=Path,
        required=True,
        help="Fixed test/anomalous image compared with every normal support.",
    )
    io_group.add_argument(
        "--gt_path",
        type=str,
        default="auto",
        help=(
            "Ground-truth mask path. Use 'auto' for MVTec AD/VisA path "
            "inference or 'none' when no mask is available."
        ),
    )
    io_group.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./visualization_results/good_bank"),
        help="Root directory for the anomaly-specific output folder.",
    )
    io_group.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output folder name; inferred from the test image otherwise.",
    )
    io_group.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category name; inferred from dataset paths otherwise.",
    )
    io_group.add_argument(
        "--recursive",
        action="store_true",
        help="Search good_dir recursively instead of using its direct children.",
    )
    io_group.add_argument(
        "--support_ids",
        nargs="*",
        default=None,
        help="Optional support stems to process, e.g. 000 017 102.",
    )
    io_group.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start position in the naturally sorted support list.",
    )
    io_group.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of support images processed in this invocation.",
    )
    io_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute outputs that already exist.",
    )
    io_group.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop on the first failed support instead of recording and continuing.",
    )
    io_group.add_argument(
        "--save_raw_maps",
        action="store_true",
        help="Save raw and normalized score arrays for every support as NPZ.",
    )

    model_group = parser.add_argument_group("Feature extraction")
    model_group.add_argument(
        "--model_ckpt",
        type=str,
        default="./facebook/dinov2-with-registers-base",
        help="Local path or Hugging Face identifier of the DINOv2 backbone.",
    )
    model_group.add_argument(
        "--image_res",
        type=int,
        default=672,
        help="Square input resolution.",
    )
    model_group.add_argument(
        "--layers",
        type=str,
        default="-4,-5,-6,-7,-8",
        help="Comma-separated hidden-state indices.",
    )
    model_group.add_argument(
        "--agg_method",
        choices=["mean", "concat", "group"],
        default="mean",
        help="Multi-layer feature aggregation method.",
    )
    model_group.add_argument(
        "--grouped_layers",
        type=str,
        default=None,
        help="Layer groups for group aggregation, e.g. '-1,-2:-3,-4'.",
    )
    model_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Feature-extraction batch size.",
    )
    model_group.add_argument(
        "--docrop",
        action="store_true",
        help="Apply the same center-crop route as main.py.",
    )
    model_group.add_argument(
        "--use_clahe",
        action="store_true",
        help="Apply CLAHE before feature extraction.",
    )
    model_group.add_argument(
        "--dino_saliency_layer",
        type=int,
        default=6,
        help="Attention layer requested by the shared extractor.",
    )

    support_group = parser.add_argument_group("Independent one-shot modeling")
    support_group.add_argument(
        "--aug_count",
        type=int,
        default=30,
        help="Number of augmented views generated from each support image.",
    )
    support_group.add_argument(
        "--aug_list",
        nargs="+",
        default=["rotate"],
        choices=["hflip", "vflip", "rotate", "color_jitter", "affine"],
        help="Support-image augmentations.",
    )
    support_group.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Disable support augmentation.",
    )
    support_group.add_argument(
        "--no_aug_categories",
        nargs="*",
        default=["transistor"],
        help="Categories for which benchmark-compatible augmentation is disabled.",
    )
    support_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed reset before every support, making each result independent "
            "of support ordering and resume position."
        ),
    )

    score_group = parser.add_argument_group("Subspace, OTM, and COTA")
    score_group.add_argument(
        "--variant",
        choices=single.SUPPORTED_VARIANTS,
        default="cota",
        help="Method used to generate each heatmap.",
    )
    score_group.add_argument(
        "--pca_ev",
        type=float,
        default=0.99,
        help="Explained-variance ratio retained by PCA.",
    )
    score_group.add_argument(
        "--pca_dim",
        type=int,
        default=None,
        help="Fixed PCA dimension; overrides pca_ev when specified.",
    )
    score_group.add_argument(
        "--whiten",
        action="store_true",
        help="Preserve the whitening option used by PCAModel.",
    )
    score_group.add_argument(
        "--drop_k",
        type=int,
        default=0,
        help="Leading retained PCA directions excluded from scoring.",
    )
    score_group.add_argument(
        "--otm_bank_size",
        type=int,
        default=8192,
        help="Maximum number of normal prototypes retained by OTM/COTA.",
    )
    score_group.add_argument(
        "--otm_bank_chunk",
        type=int,
        default=2048,
        help="Prototype chunk size for nearest-neighbor scoring.",
    )
    score_group.add_argument(
        "--otm_bank_seed",
        type=int,
        default=None,
        help="Prototype sampling seed; defaults to seed.",
    )

    vis_group = parser.add_argument_group("Visualization")
    vis_group.add_argument(
        "--cmap",
        choices=["paper", "standard", "jet_r", "jet"],
        default="paper",
        help=(
            "paper/jet_r: low score warm and high score cool; "
            "standard/jet: high score warm."
        ),
    )
    vis_group.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Opacity of the heatmap over the test image.",
    )
    vis_group.add_argument(
        "--mask_mode",
        choices=["otsu", "percentile", "none"],
        default="otsu",
        help="Region used by the overlay image; the pure heatmap is unaffected.",
    )
    vis_group.add_argument(
        "--mask_percentile",
        type=float,
        default=70.0,
        help="Overlay threshold percentile when mask_mode=percentile.",
    )
    vis_group.add_argument(
        "--morph_kernel",
        type=int,
        default=5,
        help="Opening/dilation kernel for Otsu overlays; 0 disables morphology.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.good_dir.expanduser().is_dir():
        raise NotADirectoryError(f"good directory does not exist: {args.good_dir}")
    if not args.anomaly_path.expanduser().is_file():
        raise FileNotFoundError(f"test image does not exist: {args.anomaly_path}")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.image_res <= 0 or args.batch_size <= 0:
        raise ValueError("image_res and batch_size must be positive.")
    if args.aug_count < 0:
        raise ValueError("--aug_count must be non-negative.")
    if args.otm_bank_size <= 0 or args.otm_bank_chunk <= 0:
        raise ValueError("OTM bank size and chunk size must be positive.")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay_alpha must lie in [0, 1].")
    if not 0.0 <= args.mask_percentile <= 100.0:
        raise ValueError("--mask_percentile must lie in [0, 100].")
    if args.morph_kernel < 0:
        raise ValueError("--morph_kernel must be non-negative.")
    if args.agg_method == "group" and not args.grouped_layers:
        raise ValueError("--grouped_layers is required for group aggregation.")


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_support_paths(args: argparse.Namespace) -> List[Path]:
    root = args.good_dir.expanduser().resolve()
    candidates = root.rglob("*") if args.recursive else root.iterdir()
    paths = sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )
    if args.support_ids:
        requested = {Path(value).stem for value in args.support_ids}
        paths = [path for path in paths if path.stem in requested]
        found = {path.stem for path in paths}
        missing = sorted(requested - found)
        if missing:
            logging.warning("Requested support IDs were not found: %s", missing)

    paths = paths[args.start_index :]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError("No support images matched the requested selection.")

    ids = [single.safe_name(path.stem) for path in paths]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(
            "Support filenames must have unique stems. Duplicates: "
            + ", ".join(duplicates)
        )
    return paths


def extract_query_once(
    query_image,
    extractor,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    layers = single.parse_layer_indices(args.layers)
    grouped_layers = (
        single.parse_grouped_layers(args.grouped_layers)
        if args.agg_method == "group"
        else []
    )
    tokens, grid_size, _ = extractor.extract_tokens(
        [query_image],
        args.image_res,
        layers,
        args.agg_method,
        grouped_layers,
        args.docrop,
        use_clahe=args.use_clahe,
        dino_saliency_layer=args.dino_saliency_layer,
    )
    h_p, w_p = grid_size
    return (
        np.asarray(tokens.reshape(h_p * w_p, tokens.shape[-1])),
        (int(h_p), int(w_p)),
    )


def score_cached_query(
    query_tokens: np.ndarray,
    query_grid: Tuple[int, int],
    pca_params: Dict,
    otm_params: Optional[Dict],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, float]:
    h_p, w_p = query_grid
    image_token_scores, pixel_token_scores = single.calculate_task_scores(
        query_tokens,
        pca_params,
        otm_params,
        args.variant,
        score_method="reconstruction",
        drop_k=args.drop_k,
        bank_chunk=args.otm_bank_chunk,
    )
    image_map = single.post_process_map(
        image_token_scores.reshape(h_p, w_p), args.image_res
    )
    raw_map = np.asarray(
        single.post_process_map(
            pixel_token_scores.reshape(h_p, w_p), args.image_res
        ),
        dtype=np.float32,
    )
    normalized_map = np.asarray(single.min_max_norm(raw_map), dtype=np.float32)
    image_score = float(single.topk_mean(image_map, frac=0.01))
    return raw_map, normalized_map, image_score


def read_existing_summary(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            row["support_id"]: row
            for row in csv.DictReader(file)
            if row.get("support_id")
        }


def write_summary(path: Path, rows_by_id: Dict[str, Dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for support_id in sorted(
            rows_by_id,
            key=lambda value: [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", value)
            ],
        ):
            row = {
                field: rows_by_id[support_id].get(field, "")
                for field in SUMMARY_FIELDS
            }
            writer.writerow(row)
    temporary.replace(path)


def clear_iteration_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_run_config(
    args: argparse.Namespace,
    good_paths: Sequence[Path],
    anomaly_path: Path,
    gt_path: Optional[Path],
    category: Optional[str],
    query_grid: Tuple[int, int],
) -> Dict:
    return {
        "good_dir": str(args.good_dir.expanduser().resolve()),
        "number_of_selected_supports": len(good_paths),
        "selected_support_ids": [path.stem for path in good_paths],
        "anomaly_path": str(anomaly_path),
        "gt_path": str(gt_path) if gt_path is not None else None,
        "category": category,
        "model_ckpt": args.model_ckpt,
        "image_res": args.image_res,
        "layers": args.layers,
        "agg_method": args.agg_method,
        "query_grid": list(query_grid),
        "variant": args.variant,
        "seed_reset_per_support": args.seed,
        "pca_ev": args.pca_ev,
        "pca_dim": args.pca_dim,
        "aug_count": args.aug_count,
        "aug_list": list(args.aug_list),
        "otm_bank_size": args.otm_bank_size,
        "otm_bank_chunk": args.otm_bank_chunk,
        "otm_bank_seed": (
            args.seed if args.otm_bank_seed is None else args.otm_bank_seed
        ),
        "colormap": args.cmap,
        "colormap_orientation": (
            "low score -> warm; high score -> cool"
            if args.cmap in {"paper", "jet_r"}
            else "low score -> cool; high score -> warm"
        ),
        "overlay_mask_mode": args.mask_mode,
        "normalization": (
            "Each heatmap is independently min-max normalized for visualization. "
            "Use summary.csv raw scores for quantitative support comparison."
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args.variants = [args.variant]

    good_paths = collect_support_paths(args)
    anomaly_path = args.anomaly_path.expanduser().resolve()
    category = (
        args.category
        or single.infer_category(good_paths[0])
        or single.infer_category(anomaly_path)
    )
    gt_path = single.resolve_gt_path(args.gt_path, anomaly_path)
    if args.gt_path.lower() == "auto" and gt_path is None:
        logging.warning(
            "No GT mask was inferred; gt_overlay.png will be marked unavailable."
        )

    inferred_name = single.make_output_name(None, category, anomaly_path)
    output_name = single.safe_name(args.output_name or inferred_name)
    output_dir = args.output_dir.expanduser().resolve() / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows_by_id = read_existing_summary(summary_path)

    single.set_reproducible_seed(args.seed)
    query_image = single.load_rgb(anomaly_path)
    display_image = single.prepare_display_image(
        query_image, args.image_res, args.docrop
    )
    gt_mask = single.load_gt_mask(gt_path, args.image_res, args.docrop)
    gt_panel = single.make_gt_panel(display_image, gt_mask)
    single.save_png(display_image, output_dir / "test_image.png")
    single.save_png(gt_panel, output_dir / "gt_overlay.png")

    logging.info("Loading backbone once: %s", args.model_ckpt)
    extractor = single.FeatureExtractor(args.model_ckpt)
    query_tokens, query_grid = extract_query_once(query_image, extractor, args)
    config = build_run_config(
        args,
        good_paths,
        anomaly_path,
        gt_path,
        category,
        query_grid,
    )
    with (output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    logging.info(
        "Batch visualization starts | supports=%d | variant=%s | output=%s",
        len(good_paths),
        args.variant,
        output_dir,
    )
    completed = 0
    skipped = 0
    failed = 0

    for index, support_path in enumerate(good_paths, start=1):
        support_id = single.safe_name(support_path.stem)
        heatmap_path = output_dir / f"{support_id}_heatmap.png"
        overlay_path = output_dir / f"{support_id}_overlay.png"
        raw_path = output_dir / f"{support_id}_map.npz"
        expected_paths = [heatmap_path, overlay_path]
        if args.save_raw_maps:
            expected_paths.append(raw_path)

        if not args.overwrite and all(path.is_file() for path in expected_paths):
            skipped += 1
            logging.info(
                "[%d/%d] support=%s already exists; skipped.",
                index,
                len(good_paths),
                support_id,
            )
            if support_id not in rows_by_id:
                rows_by_id[support_id] = {
                    "support_id": support_id,
                    "support_path": str(support_path),
                    "status": "skipped_existing",
                    "variant": args.variant,
                    "heatmap_file": heatmap_path.name,
                    "overlay_file": overlay_path.name,
                    "raw_map_file": raw_path.name if args.save_raw_maps else "",
                }
                write_summary(summary_path, rows_by_id)
            continue

        start_time = time.perf_counter()
        pca_params = None
        otm_params = None
        raw_map = None
        normalized_map = None
        try:
            logging.info(
                "[%d/%d] support=%s | %s",
                index,
                len(good_paths),
                support_id,
                support_path,
            )
            # Reset before every independent support.  Results therefore do not
            # depend on list ordering, slicing, failure recovery, or resume.
            single.set_reproducible_seed(args.seed)
            support_image = single.load_rgb(support_path)
            (
                pca_params,
                otm_params,
                _support_grid,
                feature_dim,
                effective_aug_count,
            ) = single.fit_one_shot_models(
                support_image,
                extractor,
                args,
                category,
            )
            if feature_dim != query_tokens.shape[1]:
                raise RuntimeError(
                    "Support/query feature dimensions differ: "
                    f"{feature_dim} vs {query_tokens.shape[1]}."
                )

            raw_map, normalized_map, image_score = score_cached_query(
                query_tokens,
                query_grid,
                pca_params,
                otm_params,
                args,
            )
            heatmap = single.colorize_score(normalized_map, args.cmap)
            overlay = single.blend_heatmap(
                display_image,
                normalized_map,
                args.cmap,
                args.overlay_alpha,
                args.mask_mode,
                args.mask_percentile,
                args.morph_kernel,
            )
            single.save_png(heatmap, heatmap_path)
            single.save_png(overlay, overlay_path)
            if args.save_raw_maps:
                np.savez_compressed(
                    raw_path,
                    raw=raw_map,
                    normalized=normalized_map,
                )

            elapsed = time.perf_counter() - start_time
            rows_by_id[support_id] = {
                "support_id": support_id,
                "support_path": str(support_path),
                "status": "success",
                "elapsed_seconds": f"{elapsed:.4f}",
                "variant": args.variant,
                "image_score_top_1_percent": f"{image_score:.10g}",
                "pca_retained_components": int(pca_params["k"]),
                "feature_dimension": int(feature_dim),
                "effective_augmentation_count": int(effective_aug_count),
                "otm_bank_size": (
                    int(otm_params["bank_size"])
                    if otm_params is not None
                    else 0
                ),
                "raw_map_min": f"{float(np.min(raw_map)):.10g}",
                "raw_map_max": f"{float(np.max(raw_map)):.10g}",
                "heatmap_file": heatmap_path.name,
                "overlay_file": overlay_path.name,
                "raw_map_file": raw_path.name if args.save_raw_maps else "",
                "error": "",
            }
            completed += 1
            write_summary(summary_path, rows_by_id)
            logging.info(
                "[%d/%d] support=%s completed in %.2f s.",
                index,
                len(good_paths),
                support_id,
                elapsed,
            )
        except Exception as error:
            elapsed = time.perf_counter() - start_time
            failed += 1
            rows_by_id[support_id] = {
                "support_id": support_id,
                "support_path": str(support_path),
                "status": "failed",
                "elapsed_seconds": f"{elapsed:.4f}",
                "variant": args.variant,
                "heatmap_file": heatmap_path.name,
                "overlay_file": overlay_path.name,
                "raw_map_file": raw_path.name if args.save_raw_maps else "",
                "error": f"{type(error).__name__}: {error}",
            }
            write_summary(summary_path, rows_by_id)
            logging.exception(
                "[%d/%d] support=%s failed.",
                index,
                len(good_paths),
                support_id,
            )
            if args.fail_fast:
                raise
        finally:
            # Drop the per-support PCA parameters and OTM GPU bank cache while
            # retaining only the backbone and the fixed query tokens.
            pca_params = None
            otm_params = None
            raw_map = None
            normalized_map = None
            support_image = None
            heatmap = None
            overlay = None
            clear_iteration_memory()

    logging.info(
        "All selected supports finished | completed=%d | skipped=%d | "
        "failed=%d | output=%s",
        completed,
        skipped,
        failed,
        output_dir,
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()