#!/usr/bin/env python3
"""Generate paper-ready one-shot qualitative results for OTM/COTA-SubspaceAD.

The tool is additive: it imports the existing feature extractor, PCA model,
OTM bank, COTA calibration, and post-processing functions without modifying
the benchmark pipeline.  A user-selected normal image is the complete
one-shot support set; a user-selected test image is scored by the baseline
SubspaceAD Q residual, OTM, and COTA in one run.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from subspacead.config import parse_grouped_layers, parse_layer_indices
from subspacead.core.extractor import FeatureExtractor
from subspacead.core.otm import (
    build_otm_bank,
    calculate_task_scores,
    fit_cota_calibration,
)
from subspacead.core.pca import PCAModel
from subspacead.data.transforms import get_augmentation_transform
from subspacead.post_process.scoring import post_process_map
from subspacead.utils.common import min_max_norm, topk_mean


VARIANT_TITLES = {
    "baseline": "SubspaceAD (Q)",
    "tangent": "Tangent (T)",
    "otm_full": "OTM",
    "cota": "COTA (Ours)",
    "cota_safe": "COTA-Safe",
}
SUPPORTED_VARIANTS = tuple(VARIANT_TITLES)
CALIBRATED_VARIANTS = {"cota", "cota_safe"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a one-shot normal subspace from one support image and produce "
            "paper-ready SubspaceAD/OTM/COTA anomaly-map comparisons."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    io_group = parser.add_argument_group("Input and output")
    io_group.add_argument(
        "--normal_path",
        type=Path,
        required=True,
        help="Path to the single normal support image.",
    )
    io_group.add_argument(
        "--anomaly_path",
        type=Path,
        required=True,
        help="Path to the test image to visualize.",
    )
    io_group.add_argument(
        "--gt_path",
        type=str,
        default="auto",
        help=(
            "Ground-truth mask path. Use 'auto' for MVTec AD/VisA path "
            "inference or 'none' to omit the mask."
        ),
    )
    io_group.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./visualization_results"),
        help="Root directory for generated qualitative results.",
    )
    io_group.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional sample-directory name. A name is inferred when omitted.",
    )
    io_group.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional category label displayed on the left of the comparison.",
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
        help="Square input resolution used by the benchmark pipeline.",
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
        help="Apply the same resize-and-center-crop route as main.py.",
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

    support_group = parser.add_argument_group("One-shot support modeling")
    support_group.add_argument(
        "--aug_count",
        type=int,
        default=30,
        help="Number of augmented views generated from the one support image.",
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
        help="Disable all support-image augmentation.",
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
        help="Random seed used by augmentation and prototype sampling.",
    )

    score_group = parser.add_argument_group("Subspace, OTM, and COTA")
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
        help="Number of leading retained PCA directions excluded from scoring.",
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
    score_group.add_argument(
        "--variants",
        nargs="+",
        choices=SUPPORTED_VARIANTS,
        default=["baseline", "otm_full", "cota"],
        help="Anomaly-map columns included after Image and GT.",
    )

    vis_group = parser.add_argument_group("Paper visualization")
    vis_group.add_argument(
        "--cmap",
        choices=["paper", "standard", "jet_r", "jet"],
        default="paper",
        help=(
            "paper/jet_r: low score warm and high score cool; "
            "standard/jet: low score cool and high score warm."
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
        help="Region used for overlay; scores are never altered.",
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
        help="Opening/dilation kernel for an Otsu overlay mask; 0 disables it.",
    )
    vis_group.add_argument(
        "--panel_size",
        type=int,
        default=None,
        help="Square panel size in pixels; defaults to image_res.",
    )
    vis_group.add_argument(
        "--no_raw_maps",
        action="store_true",
        help="Do not save raw and normalized anomaly arrays as NPZ.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("normal support image", args.normal_path),
        ("test image", args.anomaly_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.image_res <= 0:
        raise ValueError("--image_res must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
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
    if args.panel_size is not None and args.panel_size <= 0:
        raise ValueError("--panel_size must be positive.")
    if args.agg_method == "group" and not args.grouped_layers:
        raise ValueError("--grouped_layers is required when --agg_method=group.")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def infer_category(path: Path) -> Optional[str]:
    parts = path.resolve().parts
    for marker in ("train", "test", "test_public", "validation"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return parts[index - 1]
    return None


def _path_from_parts(parts: Sequence[str]) -> Path:
    if not parts:
        return Path(".")
    return Path(parts[0], *parts[1:])


def infer_gt_path(anomaly_path: Path) -> Optional[Path]:
    """Infer common MVTec AD, MVTec AD 2, and VisA mask layouts."""

    resolved = anomaly_path.resolve()
    parts = resolved.parts
    marker = None
    for candidate in ("test", "test_public"):
        if candidate in parts:
            marker = candidate
            break
    if marker is None:
        return None

    marker_index = parts.index(marker)
    if marker_index + 1 >= len(parts):
        return None
    category_root = _path_from_parts(parts[:marker_index])
    defect = parts[marker_index + 1]
    stem = resolved.stem

    candidates = []
    if marker == "test_public":
        candidates.extend(
            [
                category_root
                / "test_public"
                / "ground_truth"
                / defect
                / f"{stem}_mask.png",
                category_root
                / "test_public"
                / "ground_truth"
                / defect
                / f"{stem}.png",
            ]
        )
    candidates.extend(
        [
            category_root / "ground_truth" / defect / f"{stem}_mask.png",
            category_root / "ground_truth" / defect / f"{stem}.png",
            category_root / "ground_truth" / "bad" / f"{stem}.png",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_gt_path(value: str, anomaly_path: Path) -> Optional[Path]:
    lowered = value.strip().lower()
    if lowered in {"none", "null", "no", ""}:
        return None
    if lowered == "auto":
        return infer_gt_path(anomaly_path)
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"ground-truth mask does not exist: {path}")
    return path.resolve()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    return cleaned or "sample"


def make_output_name(
    requested: Optional[str], category: Optional[str], anomaly_path: Path
) -> str:
    if requested:
        return safe_name(requested)
    defect = anomaly_path.parent.name
    fields = [field for field in (category, defect, anomaly_path.stem) if field]
    return safe_name("__".join(fields))


def create_support_generator(
    support_image: Image.Image,
    extractor: FeatureExtractor,
    feature_dim: int,
    tokens_per_image: int,
    args: argparse.Namespace,
    layers: Sequence[int],
    grouped_layers: Sequence[Sequence[int]],
    augmentation,
):
    """Return benchmark-compatible PCA and source-aware OTM generators."""

    def generate(include_source_ids: bool = False):
        images = [support_image]
        if augmentation is not None:
            images.extend(augmentation(support_image) for _ in range(args.aug_count))

        for start in range(0, len(images), args.batch_size):
            batch = images[start : start + args.batch_size]
            tokens, _, _ = extractor.extract_tokens(
                batch,
                args.image_res,
                list(layers),
                args.agg_method,
                list(grouped_layers),
                args.docrop,
                use_clahe=args.use_clahe,
                dino_saliency_layer=args.dino_saliency_layer,
            )
            flat = np.asarray(tokens.reshape(-1, feature_dim))
            if include_source_ids:
                source_ids = np.zeros(len(flat), dtype=np.int64)
                yield flat, source_ids
            else:
                yield flat

    return (
        lambda: generate(False),
        lambda: generate(True),
        tokens_per_image,
    )


def fit_one_shot_models(
    support_image: Image.Image,
    extractor: FeatureExtractor,
    args: argparse.Namespace,
    category: Optional[str],
) -> Tuple[Dict, Optional[Dict], Tuple[int, int], int, int]:
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers)
        if args.agg_method == "group"
        else []
    )

    temp_tokens, grid_size, _ = extractor.extract_tokens(
        [support_image],
        args.image_res,
        layers,
        args.agg_method,
        grouped_layers,
        args.docrop,
        use_clahe=args.use_clahe,
        dino_saliency_layer=args.dino_saliency_layer,
    )
    h_p, w_p = grid_size
    feature_dim = int(temp_tokens.shape[-1])
    tokens_per_image = int(h_p * w_p)

    augmentation_disabled = (
        args.disable_augmentation
        or args.aug_count == 0
        or category in set(args.no_aug_categories)
    )
    augmentation = None
    if not augmentation_disabled:
        augmentation = get_augmentation_transform(args.aug_list, args.image_res)
        if not getattr(augmentation, "transforms", None):
            augmentation = None

    effective_aug_count = args.aug_count if augmentation is not None else 0
    total_images = 1 + effective_aug_count
    total_tokens = total_images * tokens_per_image
    num_batches = math.ceil(total_images / args.batch_size)
    logging.info(
        "One-shot support | category=%s | original=1 | augmentations=%d | "
        "tokens/image=%d | total tokens=%d | feature dimension=%d",
        category or "unknown",
        effective_aug_count,
        tokens_per_image,
        total_tokens,
        feature_dim,
    )

    pca_generator, otm_generator, _ = create_support_generator(
        support_image,
        extractor,
        feature_dim,
        tokens_per_image,
        args,
        layers,
        grouped_layers,
        augmentation,
    )
    pca_model = PCAModel(k=args.pca_dim, ev=args.pca_ev, whiten=args.whiten)
    pca_params = pca_model.fit(
        pca_generator,
        feature_dim=feature_dim,
        total_tokens=total_tokens,
        num_batches=num_batches,
    )

    needs_bank = any(variant != "baseline" for variant in args.variants)
    otm_params = None
    if needs_bank:
        bank_seed = args.seed if args.otm_bank_seed is None else args.otm_bank_seed
        otm_params = build_otm_bank(
            otm_generator,
            pca_params,
            variant="cota",
            bank_size=args.otm_bank_size,
            seed=bank_seed,
            drop_k=args.drop_k,
        )
        if any(variant in CALIBRATED_VARIANTS for variant in args.variants):
            otm_params = fit_cota_calibration(
                otm_params,
                pca_params,
                score_method="reconstruction",
                drop_k=args.drop_k,
                bank_chunk=args.otm_bank_chunk,
            )
    return pca_params, otm_params, grid_size, feature_dim, effective_aug_count


def score_query(
    query_image: Image.Image,
    extractor: FeatureExtractor,
    pca_params: Dict,
    otm_params: Optional[Dict],
    args: argparse.Namespace,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Tuple[int, int]]:
    layers = parse_layer_indices(args.layers)
    grouped_layers = (
        parse_grouped_layers(args.grouped_layers)
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
    flat = tokens.reshape(h_p * w_p, tokens.shape[-1])

    raw_maps: Dict[str, np.ndarray] = {}
    normalized_maps: Dict[str, np.ndarray] = {}
    image_scores: Dict[str, float] = {}
    for variant in args.variants:
        image_token_scores, pixel_token_scores = calculate_task_scores(
            flat,
            pca_params,
            otm_params,
            variant,
            score_method="reconstruction",
            drop_k=args.drop_k,
            bank_chunk=args.otm_bank_chunk,
        )
        image_map = post_process_map(
            image_token_scores.reshape(h_p, w_p), args.image_res
        )
        pixel_map = post_process_map(
            pixel_token_scores.reshape(h_p, w_p), args.image_res
        )
        raw_maps[variant] = np.asarray(pixel_map, dtype=np.float32)
        normalized_maps[variant] = np.asarray(
            min_max_norm(pixel_map), dtype=np.float32
        )
        image_scores[variant] = topk_mean(image_map, frac=0.01)
    return raw_maps, normalized_maps, image_scores, grid_size


def _resampling(name: str):
    if hasattr(Image, "Resampling"):
        return getattr(Image.Resampling, name)
    return getattr(Image, name)


def prepare_display_image(
    image: Image.Image, image_res: int, docrop: bool
) -> np.ndarray:
    if docrop:
        resize_res = int(image_res / 0.875)
        resized = image.resize(
            (resize_res, resize_res), _resampling("BICUBIC")
        )
        left = max(0, (resize_res - image_res) // 2)
        upper = max(0, (resize_res - image_res) // 2)
        resized = resized.crop(
            (left, upper, left + image_res, upper + image_res)
        )
    else:
        resized = image.resize(
            (image_res, image_res), _resampling("BICUBIC")
        )
    return np.asarray(resized.convert("RGB"), dtype=np.uint8)


def load_gt_mask(
    gt_path: Optional[Path], image_res: int, docrop: bool
) -> Optional[np.ndarray]:
    if gt_path is None:
        return None
    with Image.open(gt_path) as image:
        mask = ImageOps.exif_transpose(image).convert("L")
        if docrop:
            resize_res = int(image_res / 0.875)
            mask = mask.resize(
                (resize_res, resize_res), _resampling("NEAREST")
            )
            left = max(0, (resize_res - image_res) // 2)
            upper = max(0, (resize_res - image_res) // 2)
            mask = mask.crop(
                (left, upper, left + image_res, upper + image_res)
            )
        else:
            mask = mask.resize(
                (image_res, image_res), _resampling("NEAREST")
            )
    return (np.asarray(mask) > 0).astype(np.uint8)


def colorize_score(score_norm: np.ndarray, cmap: str) -> np.ndarray:
    """Apply explicit JET orientation and return an RGB heatmap."""

    score_u8 = np.clip(np.rint(score_norm * 255.0), 0, 255).astype(np.uint8)
    paper_orientation = cmap in {"paper", "jet_r"}
    color_input = 255 - score_u8 if paper_orientation else score_u8
    heatmap_bgr = cv2.applyColorMap(color_input, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def make_overlay_mask(
    score_norm: np.ndarray,
    mode: str,
    percentile: float,
    morph_kernel: int,
) -> np.ndarray:
    score_u8 = np.clip(np.rint(score_norm * 255.0), 0, 255).astype(np.uint8)
    if mode == "none":
        return np.ones(score_u8.shape, dtype=bool)
    if mode == "percentile":
        threshold = float(np.percentile(score_norm, percentile))
        return score_norm >= threshold

    _, binary = cv2.threshold(
        score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if morph_kernel > 0:
        kernel_size = morph_kernel + (1 - morph_kernel % 2)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.dilate(binary, kernel, iterations=1)
    return binary > 0


def blend_heatmap(
    image_rgb: np.ndarray,
    score_norm: np.ndarray,
    cmap: str,
    alpha: float,
    mask_mode: str,
    mask_percentile: float,
    morph_kernel: int,
) -> np.ndarray:
    heatmap = colorize_score(score_norm, cmap)
    blended = np.clip(
        image_rgb.astype(np.float32) * (1.0 - alpha)
        + heatmap.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)
    mask = make_overlay_mask(
        score_norm, mask_mode, mask_percentile, morph_kernel
    )
    return np.where(mask[..., None], blended, image_rgb)


def make_gt_panel(
    image_rgb: np.ndarray, gt_mask: Optional[np.ndarray]
) -> np.ndarray:
    if gt_mask is None:
        panel = np.clip(image_rgb.astype(np.float32) * 0.55, 0, 255).astype(
            np.uint8
        )
        pil_panel = Image.fromarray(panel)
        draw = ImageDraw.Draw(pil_panel)
        font = load_font(max(16, image_rgb.shape[0] // 28), bold=False)
        message = "GT unavailable"
        box = draw.textbbox((0, 0), message, font=font)
        x = (image_rgb.shape[1] - (box[2] - box[0])) // 2
        y = (image_rgb.shape[0] - (box[3] - box[1])) // 2
        draw.text((x, y), message, fill=(245, 245, 245), font=font)
        return np.asarray(pil_panel)

    dimmed = np.clip(image_rgb.astype(np.float32) * 0.78, 0, 255)
    overlay_color = np.array([67, 55, 205], dtype=np.float32)
    masked = dimmed.copy()
    alpha = 0.58
    masked[gt_mask > 0] = (
        dimmed[gt_mask > 0] * (1.0 - alpha) + overlay_color * alpha
    )
    return np.clip(masked, 0, 255).astype(np.uint8)


def load_font(size: int, bold: bool = False, italic: bool = False):
    style = "BoldOblique" if bold and italic else "Bold" if bold else "Oblique" if italic else ""
    filenames = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif-{style}.ttf"
        if style
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans-{style}.ttf"
        if style
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for filename in filenames:
        try:
            return ImageFont.truetype(filename, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_panel(panel: np.ndarray, panel_size: int) -> Image.Image:
    return Image.fromarray(panel).resize(
        (panel_size, panel_size), _resampling("LANCZOS")
    )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font,
    fill=(20, 20, 20),
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    draw.text((center_x - width // 2, y), text, font=font, fill=fill)


def compose_paper_figure(
    panels: Sequence[np.ndarray],
    titles: Sequence[str],
    category: Optional[str],
    panel_size: int,
) -> Image.Image:
    gap = max(4, panel_size // 80)
    header_height = max(64, panel_size // 7)
    label_width = max(70, panel_size // 7) if category else 0
    width = label_width + len(panels) * panel_size + (len(panels) - 1) * gap
    height = header_height + panel_size
    canvas = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(20, panel_size // 20), bold=False)
    ours_font = load_font(max(20, panel_size // 20), bold=True, italic=True)

    for index, (panel, title) in enumerate(zip(panels, titles)):
        x = label_width + index * (panel_size + gap)
        canvas.paste(resize_panel(panel, panel_size), (x, header_height))
        font = ours_font if "Ours" in title else title_font
        draw_centered_text(
            draw,
            title,
            x + panel_size // 2,
            max(8, header_height // 4),
            font,
        )

    if category:
        label_font = load_font(max(20, panel_size // 18), bold=False)
        label_box = label_font.getbbox(category)
        label_w = max(1, label_box[2] - label_box[0] + 20)
        label_h = max(1, label_box[3] - label_box[1] + 20)
        label_image = Image.new("RGBA", (label_w, label_h), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((10, 8), category, font=label_font, fill=(20, 20, 20, 255))
        rotated = label_image.rotate(90, expand=True)
        x = max(0, (label_width - rotated.width) // 2)
        y = header_height + max(0, (panel_size - rotated.height) // 2)
        canvas.paste(rotated, (x, y), rotated)
    return canvas


def save_png(array_or_image, path: Path, dpi: int = 300) -> None:
    image = (
        array_or_image
        if isinstance(array_or_image, Image.Image)
        else Image.fromarray(np.asarray(array_or_image, dtype=np.uint8))
    )
    image.save(path, dpi=(dpi, dpi), optimize=True)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_reproducible_seed(args.seed)

    normal_path = args.normal_path.expanduser().resolve()
    anomaly_path = args.anomaly_path.expanduser().resolve()
    category = args.category or infer_category(normal_path) or infer_category(
        anomaly_path
    )
    gt_path = resolve_gt_path(args.gt_path, anomaly_path)
    if args.gt_path.lower() == "auto" and gt_path is None:
        logging.warning(
            "No ground-truth mask was inferred; the GT panel will be marked unavailable."
        )

    output_name = make_output_name(args.output_name, category, anomaly_path)
    sample_dir = args.output_dir.expanduser().resolve() / output_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    support_image = load_rgb(normal_path)
    query_image = load_rgb(anomaly_path)
    logging.info("Loading backbone: %s", args.model_ckpt)
    extractor = FeatureExtractor(args.model_ckpt)

    pca_params, otm_params, support_grid, feature_dim, effective_aug_count = (
        fit_one_shot_models(support_image, extractor, args, category)
    )
    raw_maps, normalized_maps, image_scores, query_grid = score_query(
        query_image, extractor, pca_params, otm_params, args
    )

    display_image = prepare_display_image(
        query_image, args.image_res, args.docrop
    )
    display_support = prepare_display_image(
        support_image, args.image_res, args.docrop
    )
    gt_mask = load_gt_mask(gt_path, args.image_res, args.docrop)
    gt_panel = make_gt_panel(display_image, gt_mask)

    overlays: Dict[str, np.ndarray] = {}
    for variant in args.variants:
        overlays[variant] = blend_heatmap(
            display_image,
            normalized_maps[variant],
            args.cmap,
            args.overlay_alpha,
            args.mask_mode,
            args.mask_percentile,
            args.morph_kernel,
        )

    panels = [display_image, gt_panel] + [
        overlays[variant] for variant in args.variants
    ]
    titles = ["Image", "GT"] + [
        VARIANT_TITLES[variant] for variant in args.variants
    ]
    panel_size = args.panel_size or args.image_res
    comparison = compose_paper_figure(
        panels, titles, category, panel_size=panel_size
    )

    comparison_path = sample_dir / "comparison.png"
    save_png(comparison, comparison_path)
    save_png(display_support, sample_dir / "support_1shot.png")
    save_png(display_image, sample_dir / "test_image.png")
    save_png(gt_panel, sample_dir / "gt_overlay.png")
    for variant in args.variants:
        save_png(overlays[variant], sample_dir / f"{variant}_overlay.png")
        heatmap = colorize_score(normalized_maps[variant], args.cmap)
        save_png(heatmap, sample_dir / f"{variant}_heatmap.png")

    if not args.no_raw_maps:
        arrays = {}
        for variant in args.variants:
            arrays[f"{variant}_raw"] = raw_maps[variant]
            arrays[f"{variant}_normalized"] = normalized_maps[variant]
        if gt_mask is not None:
            arrays["gt_mask"] = gt_mask
        np.savez_compressed(sample_dir / "anomaly_maps.npz", **arrays)

    metadata = {
        "normal_path": str(normal_path),
        "anomaly_path": str(anomaly_path),
        "gt_path": str(gt_path) if gt_path is not None else None,
        "category": category,
        "model_ckpt": args.model_ckpt,
        "image_res": args.image_res,
        "layers": args.layers,
        "agg_method": args.agg_method,
        "seed": args.seed,
        "pca_ev": args.pca_ev,
        "pca_dim": args.pca_dim,
        "pca_retained_components": int(pca_params["k"]),
        "feature_dimension": feature_dim,
        "support_grid": list(support_grid),
        "query_grid": list(query_grid),
        "effective_augmentation_count": effective_aug_count,
        "otm_bank_size": (
            int(otm_params["bank_size"]) if otm_params is not None else 0
        ),
        "otm_bank_chunk": args.otm_bank_chunk,
        "variants": list(args.variants),
        "colormap": args.cmap,
        "colormap_orientation": (
            "low score -> warm; high score -> cool"
            if args.cmap in {"paper", "jet_r"}
            else "low score -> cool; high score -> warm"
        ),
        "overlay_mask_mode": args.mask_mode,
        "image_scores_top_1_percent": {
            name: float(value) for name, value in image_scores.items()
        },
        "raw_map_ranges": {
            name: {
                "min": float(np.min(value)),
                "max": float(np.max(value)),
            }
            for name, value in raw_maps.items()
        },
    }
    with (sample_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    logging.info("Comparison saved to: %s", comparison_path)
    logging.info("All outputs saved under: %s", sample_dir)
    print(str(comparison_path))


if __name__ == "__main__":
    main()
