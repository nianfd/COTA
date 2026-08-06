"""Orthogonal-tangential modeling and COTA pixel scoring.

The original SubspaceAD score is the squared PCA reconstruction residual

    Q(x) = ||(I - C C^T)(x - mu)||^2.

OTM complements it with a nearest-normal-prototype distance in the retained
PCA coordinates

    T(x) = min_j ||C^T(x - mu) - z_j||^2.

This module keeps the published ``baseline``, ``tangent``, ``raw_knn`` and
``otm_full`` routes intact.  It additionally supports task-decoupled image and
pixel scores:

* image score map: Q + T (OTM);
* pixel score map: calibrated orthogonal/tangential evidence (COTA).

COTA uses only normal support tokens.  The prototype bank is split into two
cross-fit halves, so normal calibration queries are never matched against the
same half that contains them.  No anomaly labels or target-domain fine-tuning
are used.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # NumPy-only geometry/unit-test environment.
    torch = None


FeatureBatch = Tuple[np.ndarray, np.ndarray]
FeatureGenerator = Callable[[], Iterable[FeatureBatch]]

LEGACY_VARIANTS = {"baseline", "tangent", "raw_knn", "otm_full"}
DECOUPLED_VARIANTS = {
    "otm_decoupled",
    "calibrated_q",
    "calibrated_sum",
    "cota",
    "cota_safe",
}
COTA_CALIBRATED_VARIANTS = {
    "calibrated_q",
    "calibrated_sum",
    "cota",
    "cota_safe",
}
ALL_VARIANTS = LEGACY_VARIANTS | DECOUPLED_VARIANTS


def _effective_components(
    pca_params: Dict, drop_k: int, dtype=np.float32
) -> np.ndarray:
    if drop_k < 0:
        raise ValueError("drop_k must be non-negative.")
    retained = int(pca_params["k"])
    component_matrix = np.asarray(pca_params["components"])
    if drop_k >= retained:
        return np.empty((component_matrix.shape[0], 0), dtype=dtype)
    return np.asarray(component_matrix[:, drop_k:retained], dtype=dtype)


def _streaming_random_key_coreset(
    feature_generator: FeatureGenerator,
    representation_fn: Callable[[np.ndarray], np.ndarray],
    bank_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Uniform streaming coreset with aligned raw features and source IDs."""

    rng = np.random.default_rng(seed)
    selected_keys = np.empty((0,), dtype=np.float64)
    selected_repr = None
    selected_raw = None
    selected_source_ids = np.empty((0,), dtype=np.int64)
    seen = 0

    for features, source_ids in feature_generator():
        raw_features = np.asarray(features, dtype=np.float32)
        source_ids = np.asarray(source_ids, dtype=np.int64).reshape(-1)
        if raw_features.ndim != 2:
            raise ValueError(
                "Prototype features must have shape [N, D], "
                f"got {raw_features.shape}."
            )
        if len(raw_features) != len(source_ids):
            raise ValueError(
                "Each prototype feature must have one source-image identifier."
            )
        if len(raw_features) == 0:
            continue

        representations = np.asarray(
            representation_fn(raw_features), dtype=np.float32
        )
        keys = rng.random(len(representations))
        seen += len(representations)

        if selected_repr is None:
            selected_repr = np.empty(
                (0, representations.shape[1]), dtype=np.float32
            )
            selected_raw = np.empty(
                (0, raw_features.shape[1]), dtype=np.float32
            )

        if len(selected_keys) >= bank_size:
            useful = keys < float(np.max(selected_keys))
            if not np.any(useful):
                continue
            keys = keys[useful]
            representations = representations[useful]
            raw_features = raw_features[useful]
            source_ids = source_ids[useful]

        merged_keys = np.concatenate([selected_keys, keys])
        merged_repr = np.concatenate([selected_repr, representations], axis=0)
        merged_raw = np.concatenate([selected_raw, raw_features], axis=0)
        merged_ids = np.concatenate([selected_source_ids, source_ids])

        if len(merged_keys) > bank_size:
            keep = np.argpartition(merged_keys, bank_size - 1)[:bank_size]
            selected_keys = merged_keys[keep]
            selected_repr = merged_repr[keep]
            selected_raw = merged_raw[keep]
            selected_source_ids = merged_ids[keep]
        else:
            selected_keys = merged_keys
            selected_repr = merged_repr
            selected_raw = merged_raw
            selected_source_ids = merged_ids

    if selected_repr is None or len(selected_repr) == 0:
        raise RuntimeError("No features were available to build the OTM bank.")

    order = np.argsort(selected_keys, kind="stable")
    return (
        np.ascontiguousarray(selected_repr[order], dtype=np.float32),
        np.ascontiguousarray(selected_raw[order], dtype=np.float32),
        np.ascontiguousarray(selected_source_ids[order], dtype=np.int64),
        seen,
    )


def build_otm_bank(
    feature_generator_with_ids: FeatureGenerator,
    pca_params: Dict,
    variant: str,
    bank_size: int = 8192,
    seed: int = 42,
    drop_k: int = 0,
) -> Dict:
    """Build a bounded normal prototype bank for an OTM/COTA variant."""

    if variant not in ALL_VARIANTS - {"baseline"}:
        raise ValueError(f"Cannot build a prototype bank for variant '{variant}'.")
    if bank_size <= 0:
        raise ValueError("otm_bank_size must be positive.")
    if "kpca" in pca_params:
        raise ValueError("OTM/COTA variants require standard PCA, not Kernel PCA.")

    mu = np.asarray(pca_params["mu"], dtype=np.float32)
    components = _effective_components(pca_params, drop_k, dtype=np.float32)
    if variant != "raw_knn" and components.shape[1] == 0:
        raise ValueError(
            "OTM/COTA requires at least one PCA component after drop_k."
        )

    if variant == "raw_knn":
        representation_fn = lambda x: np.asarray(x, dtype=np.float32)
        representation = "raw"
    else:
        representation_fn = (
            lambda x: (np.asarray(x, dtype=np.float32) - mu) @ components
        )
        representation = "tangent"

    bank, raw_support, source_ids, seen = _streaming_random_key_coreset(
        feature_generator_with_ids,
        representation_fn,
        bank_size,
        seed,
    )

    logging.info(
        "OTM bank built | variant=%s | representation=%s | "
        "observed_tokens=%d | retained_prototypes=%d | dimension=%d",
        variant,
        representation,
        seen,
        len(bank),
        bank.shape[1],
    )
    return {
        "variant": variant,
        "representation": representation,
        "bank": bank,
        "raw_support": raw_support,
        "source_ids": source_ids,
        "bank_size": int(len(bank)),
        "observed_tokens": int(seen),
        "seed": int(seed),
        "drop_k": int(drop_k),
        "_torch_cache": {},
    }


def _split_cross_fit_indices(source_ids: np.ndarray, seed: int):
    """Deterministically split prototypes, preferring source-image separation."""

    source_ids = np.asarray(source_ids, dtype=np.int64)
    n = len(source_ids)
    if n < 4:
        raise ValueError(
            "COTA calibration needs at least four retained normal prototypes."
        )
    rng = np.random.default_rng(seed + 104729)
    unique_ids = np.unique(source_ids)

    if len(unique_ids) >= 2:
        shuffled = rng.permutation(unique_ids)
        cut = max(1, min(len(shuffled) - 1, len(shuffled) // 2))
        ids_a = shuffled[:cut]
        in_a = np.isin(source_ids, ids_a)
        idx_a = np.flatnonzero(in_a)
        idx_b = np.flatnonzero(~in_a)
        split_unit = "source_image"
    else:
        order = rng.permutation(n)
        cut = n // 2
        idx_a = order[:cut]
        idx_b = order[cut:]
        split_unit = "prototype"

    if len(idx_a) == 0 or len(idx_b) == 0:
        raise RuntimeError("COTA cross-fit split produced an empty half.")
    return (
        np.ascontiguousarray(idx_a, dtype=np.int64),
        np.ascontiguousarray(idx_b, dtype=np.int64),
        split_unit,
    )


def _minimum_squared_distance_numpy(
    query: np.ndarray,
    bank: np.ndarray,
    bank_source_ids: np.ndarray,
    query_source_ids: Optional[np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    """Chunked NumPy squared NN distance with optional same-source exclusion."""

    if chunk_size <= 0:
        raise ValueError("otm_bank_chunk must be positive.")
    query = np.asarray(query, dtype=np.float32)
    bank = np.asarray(bank, dtype=np.float32)
    bank_source_ids = np.asarray(bank_source_ids, dtype=np.int64)
    if len(bank) == 0:
        raise ValueError("The prototype bank cannot be empty.")

    query_norm = np.sum(query * query, axis=1, keepdims=True)
    minimum = np.full(len(query), np.inf, dtype=np.float32)
    minimum_unmasked = minimum.copy()

    query_ids = None
    if query_source_ids is not None:
        query_ids = np.asarray(query_source_ids, dtype=np.int64).reshape(-1)
        if len(query_ids) != len(query):
            raise ValueError(
                "query_source_ids must contain one identifier per query token."
            )

    for start in range(0, len(bank), chunk_size):
        bank_chunk = bank[start : start + chunk_size]
        bank_norm = np.sum(bank_chunk * bank_chunk, axis=1)[None, :]
        distances = query_norm + bank_norm - 2.0 * (query @ bank_chunk.T)
        np.maximum(distances, 0.0, out=distances)
        minimum_unmasked = np.minimum(minimum_unmasked, np.min(distances, axis=1))

        if query_ids is not None:
            same_source = (
                query_ids[:, None]
                == bank_source_ids[start : start + chunk_size][None, :]
            )
            distances[same_source] = np.inf
        minimum = np.minimum(minimum, np.min(distances, axis=1))

    no_cross_image_match = ~np.isfinite(minimum)
    if np.any(no_cross_image_match):
        logging.warning(
            "No cross-image prototype was available for %d query tokens; "
            "falling back to the unmasked nearest prototype.",
            int(np.sum(no_cross_image_match)),
        )
        minimum[no_cross_image_match] = minimum_unmasked[no_cross_image_match]
    return minimum


def _minimum_squared_distance_torch(
    query,
    bank,
    bank_source_ids,
    query_source_ids: Optional[np.ndarray],
    chunk_size: int,
):
    """Torch equivalent of :func:`_minimum_squared_distance_numpy`."""

    if chunk_size <= 0:
        raise ValueError("otm_bank_chunk must be positive.")
    query = query.float()
    query_norm = torch.sum(query * query, dim=1, keepdim=True)
    minimum = torch.full(
        (query.shape[0],),
        float("inf"),
        dtype=torch.float32,
        device=query.device,
    )
    minimum_unmasked = minimum.clone()

    query_ids = None
    if query_source_ids is not None:
        ids_np = np.asarray(query_source_ids, dtype=np.int64).reshape(-1)
        if len(ids_np) != len(query):
            raise ValueError(
                "query_source_ids must contain one identifier per query token."
            )
        query_ids = torch.from_numpy(ids_np).to(device=query.device)

    for start in range(0, len(bank), chunk_size):
        bank_chunk = bank[start : start + chunk_size]
        bank_norm = torch.sum(bank_chunk * bank_chunk, dim=1).unsqueeze(0)
        distances = query_norm + bank_norm - 2.0 * (query @ bank_chunk.T)
        distances = torch.clamp_min(distances, 0.0)
        minimum_unmasked = torch.minimum(
            minimum_unmasked, torch.min(distances, dim=1).values
        )

        if query_ids is not None:
            ids_chunk = bank_source_ids[start : start + chunk_size]
            distances = distances.masked_fill(
                query_ids[:, None] == ids_chunk[None, :], float("inf")
            )
        minimum = torch.minimum(minimum, torch.min(distances, dim=1).values)

    no_cross_image_match = ~torch.isfinite(minimum)
    if torch.any(no_cross_image_match):
        logging.warning(
            "No cross-image prototype was available for %d query tokens; "
            "falling back to the unmasked nearest prototype.",
            int(no_cross_image_match.sum().item()),
        )
        minimum[no_cross_image_match] = minimum_unmasked[no_cross_image_match]
    return minimum


def _named_bank(otm_params: Dict, name: str):
    if name == "full":
        return otm_params["bank"], otm_params["source_ids"]
    calibration = otm_params.get("cota_calibration")
    if calibration is None:
        raise ValueError(
            "This variant requires COTA calibration; call "
            "fit_cota_calibration after build_otm_bank."
        )
    indices = calibration[f"indices_{name}"]
    return otm_params["bank"][indices], otm_params["source_ids"][indices]


def _cached_named_bank_tensors(otm_params: Dict, name: str, device):
    cache = otm_params.setdefault("_torch_cache", {})
    cache_key = (name, str(device))
    if cache_key not in cache:
        bank_np, ids_np = _named_bank(otm_params, name)
        cache[cache_key] = (
            torch.from_numpy(np.ascontiguousarray(bank_np, dtype=np.float32)).to(
                device=device, dtype=torch.float32
            ),
            torch.from_numpy(np.ascontiguousarray(ids_np, dtype=np.int64)).to(
                device=device
            ),
        )
    return cache[cache_key]


def _distance_to_named_bank(
    query_repr: np.ndarray,
    otm_params: Dict,
    name: str,
    query_source_ids: Optional[np.ndarray],
    chunk_size: int,
) -> np.ndarray:
    query_np = np.ascontiguousarray(query_repr, dtype=np.float32)
    if torch is None:
        bank_np, ids_np = _named_bank(otm_params, name)
        return _minimum_squared_distance_numpy(
            query_np, bank_np, ids_np, query_source_ids, chunk_size
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bank, bank_ids = _cached_named_bank_tensors(otm_params, name, device)
    query = torch.from_numpy(query_np).to(device=device)
    return (
        _minimum_squared_distance_torch(
            query, bank, bank_ids, query_source_ids, chunk_size
        )
        .detach()
        .cpu()
        .numpy()
    )


def _orthogonal_scores(
    X: np.ndarray,
    pca_params: Dict,
    score_method: str,
    drop_k: int,
) -> np.ndarray:
    from subspacead.post_process.scoring import calculate_anomaly_scores

    return np.asarray(
        calculate_anomaly_scores(X, pca_params, score_method, drop_k)
    )


def _tangent_coordinates(
    X: np.ndarray, pca_params: Dict, drop_k: int
) -> np.ndarray:
    mu = np.asarray(pca_params["mu"], dtype=np.float32)
    components = _effective_components(pca_params, drop_k, dtype=np.float32)
    return (np.asarray(X, dtype=np.float32) - mu) @ components


def _tail_evidence(values: np.ndarray, sorted_normal: np.ndarray) -> np.ndarray:
    """Smoothed upper-tail conformal surprisal ``-log(p)``."""

    values = np.asarray(values, dtype=np.float64)
    sorted_normal = np.asarray(sorted_normal, dtype=np.float64)
    if len(sorted_normal) == 0:
        raise ValueError("Normal calibration scores cannot be empty.")
    greater_equal = len(sorted_normal) - np.searchsorted(
        sorted_normal, values, side="left"
    )
    p_value = (1.0 + greater_equal) / (len(sorted_normal) + 1.0)
    return -np.log(p_value)


def _ascending_conformal_rank(
    values: np.ndarray, sorted_normal: np.ndarray
) -> np.ndarray:
    """Integer normal-calibrated rank in ``[0, N]`` (larger = anomalous)."""

    return np.searchsorted(
        np.asarray(sorted_normal, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        side="right",
    ).astype(np.float64)


def fit_cota_calibration(
    otm_params: Dict,
    pca_params: Dict,
    score_method: str = "reconstruction",
    drop_k: int = 0,
    bank_chunk: int = 2048,
) -> Dict:
    """Fit cross-normal COTA calibration distributions.

    Half A calibrates distances evaluated against bank B and vice versa.  For
    one-shot support (one source image), tokens are split deterministically.
    """

    if otm_params.get("representation") != "tangent":
        raise ValueError("COTA calibration requires a tangent-coordinate bank.")
    idx_a, idx_b, split_unit = _split_cross_fit_indices(
        otm_params["source_ids"], int(otm_params["seed"])
    )
    bank = np.asarray(otm_params["bank"], dtype=np.float32)
    ids = np.asarray(otm_params["source_ids"], dtype=np.int64)

    # A query is calibrated using the opposite half as reference.  No self
    # prototype can be used even when only one support image is available.
    tangent_a_to_b = _minimum_squared_distance_numpy(
        bank[idx_a],
        bank[idx_b],
        ids[idx_b],
        None,
        bank_chunk,
    )
    tangent_b_to_a = _minimum_squared_distance_numpy(
        bank[idx_b],
        bank[idx_a],
        ids[idx_a],
        None,
        bank_chunk,
    )
    q_normal = _orthogonal_scores(
        np.asarray(otm_params["raw_support"]),
        pca_params,
        score_method,
        drop_k,
    )

    calibration = {
        "indices_a": idx_a,
        "indices_b": idx_b,
        # Test distance to A is compared with normal B->A distances.
        "q_sorted": np.sort(np.asarray(q_normal, dtype=np.float64)),
        "tangent_to_a_sorted": np.sort(
            np.asarray(tangent_b_to_a, dtype=np.float64)
        ),
        # Test distance to B is compared with normal A->B distances.
        "tangent_to_b_sorted": np.sort(
            np.asarray(tangent_a_to_b, dtype=np.float64)
        ),
        "split_unit": split_unit,
    }
    otm_params["cota_calibration"] = calibration
    otm_params["_torch_cache"] = {}
    logging.info(
        "COTA calibrated | split_unit=%s | half_A=%d | half_B=%d | q=%d",
        split_unit,
        len(idx_a),
        len(idx_b),
        len(q_normal),
    )
    return otm_params


def calculate_task_scores(
    X: np.ndarray,
    pca_params: Dict,
    otm_params: Optional[Dict],
    variant: str,
    score_method: str = "reconstruction",
    drop_k: int = 0,
    query_source_ids: Optional[np.ndarray] = None,
    bank_chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(image_branch_token_scores, pixel_branch_token_scores)``."""

    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown OTM/COTA variant '{variant}'.")
    X_np = np.asarray(X)
    q_score = _orthogonal_scores(X_np, pca_params, score_method, drop_k)

    if variant == "baseline":
        return q_score, q_score
    if otm_params is None:
        raise ValueError(f"Variant '{variant}' requires a fitted prototype bank.")

    if variant == "raw_knn":
        raw_distance = _distance_to_named_bank(
            np.asarray(X_np, dtype=np.float32),
            otm_params,
            "full",
            query_source_ids,
            bank_chunk,
        ).astype(X_np.dtype, copy=False)
        return raw_distance, raw_distance

    tangent_coords = _tangent_coordinates(X_np, pca_params, drop_k)
    if variant in COTA_CALIBRATED_VARIANTS:
        calibration = otm_params.get("cota_calibration")
        if calibration is None:
            raise ValueError(
                f"Variant '{variant}' requires fit_cota_calibration()."
            )
        tangent_a = _distance_to_named_bank(
            tangent_coords,
            otm_params,
            "a",
            query_source_ids,
            bank_chunk,
        )
        tangent_b = _distance_to_named_bank(
            tangent_coords,
            otm_params,
            "b",
            query_source_ids,
            bank_chunk,
        )
        # Keep the image branch identical to the original full-bank OTM route.
        # This matters in batched zero-shot: a split half can contain only the
        # query image and would otherwise fall back to an unmasked self-match.
        if query_source_ids is None:
            tangent_full = np.minimum(tangent_a, tangent_b)
        else:
            tangent_full = _distance_to_named_bank(
                tangent_coords,
                otm_params,
                "full",
                query_source_ids,
                bank_chunk,
            )
    else:
        tangent_full = _distance_to_named_bank(
            tangent_coords,
            otm_params,
            "full",
            query_source_ids,
            bank_chunk,
        )

    tangent_full = tangent_full.astype(q_score.dtype, copy=False)
    if variant == "tangent":
        return tangent_full, tangent_full

    image_score = (q_score + tangent_full).astype(X_np.dtype, copy=False)
    if variant == "otm_full":
        return image_score, image_score
    if variant == "otm_decoupled":
        return image_score, q_score

    e_q = _tail_evidence(q_score, calibration["q_sorted"])
    e_a = _tail_evidence(
        tangent_a, calibration["tangent_to_a_sorted"]
    )
    e_b = _tail_evidence(
        tangent_b, calibration["tangent_to_b_sorted"]
    )

    if variant == "calibrated_q":
        pixel_score = e_q
    elif variant == "calibrated_sum":
        pixel_score = e_q + 0.5 * (e_a + e_b)
    elif variant == "cota":
        # Tangential evidence can only contribute when both independent
        # prototype halves and the orthogonal residual agree.
        pixel_score = e_q + np.minimum.reduce([e_q, e_a, e_b])
    else:  # cota_safe
        # Lexicographic conformal refinement.  Distinct Q conformal bins can
        # never be inverted; tangent agreement only orders Q-bin ties.
        q_rank = _ascending_conformal_rank(q_score, calibration["q_sorted"])
        rank_a = _ascending_conformal_rank(
            tangent_a, calibration["tangent_to_a_sorted"]
        ) / max(1, len(calibration["tangent_to_a_sorted"]))
        rank_b = _ascending_conformal_rank(
            tangent_b, calibration["tangent_to_b_sorted"]
        ) / max(1, len(calibration["tangent_to_b_sorted"]))
        tie_break = np.minimum(rank_a, rank_b)
        pixel_score = q_rank + (1.0 - 1e-6) * tie_break

    return (
        image_score.astype(X_np.dtype, copy=False),
        np.asarray(pixel_score).astype(X_np.dtype, copy=False),
    )


def calculate_otm_scores(
    X: np.ndarray,
    pca_params: Dict,
    otm_params: Optional[Dict],
    variant: str,
    score_method: str = "reconstruction",
    drop_k: int = 0,
    query_source_ids: Optional[np.ndarray] = None,
    bank_chunk: int = 2048,
) -> np.ndarray:
    """Backward-compatible single-map API.

    Legacy variants are bit-for-bit routed as before.  New task-decoupled
    variants return their pixel branch; the benchmark entry point uses
    :func:`calculate_task_scores` to retain both maps.
    """

    _, pixel_scores = calculate_task_scores(
        X,
        pca_params,
        otm_params,
        variant,
        score_method=score_method,
        drop_k=drop_k,
        query_source_ids=query_source_ids,
        bank_chunk=bank_chunk,
    )
    return pixel_scores
