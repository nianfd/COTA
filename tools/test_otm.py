#!/usr/bin/env python3
"""Fast synthetic checks for baseline compatibility and OTM geometry."""

from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from subspacead.core.otm import (
    build_otm_bank,
    calculate_otm_scores,
    calculate_task_scores,
    fit_cota_calibration,
)

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    class Tensor:
        pass

    torch_stub.Tensor = Tensor
    torch_stub.is_tensor = lambda _: False
    sys.modules["torch"] = torch_stub

from subspacead.post_process.scoring import calculate_anomaly_scores


def make_pca(rng, feature_dim=7, retained_dim=4):
    basis, _ = np.linalg.qr(rng.normal(size=(feature_dim, feature_dim)))
    return {
        "mu": rng.normal(size=feature_dim),
        "components": basis[:, :retained_dim],
        "eigvals": np.linspace(2.0, 1.0, retained_dim),
        "k": retained_dim,
        "eps": 1e-6,
    }


def test_baseline_is_exact():
    rng = np.random.default_rng(4)
    pca = make_pca(rng)
    query = rng.normal(size=(19, 7)).astype(np.float32)
    original = calculate_anomaly_scores(
        query, pca, method="reconstruction", drop_k=0
    )
    routed = calculate_otm_scores(
        query,
        pca,
        None,
        "baseline",
        score_method="reconstruction",
        drop_k=0,
    )
    np.testing.assert_array_equal(original, routed)


def test_pythagorean_identity():
    rng = np.random.default_rng(8)
    pca = make_pca(rng, feature_dim=6, retained_dim=3)
    support = rng.normal(size=(13, 6)).astype(np.float32)
    support_ids = np.arange(len(support), dtype=np.int64)

    def generator():
        yield support[:7], support_ids[:7]
        yield support[7:], support_ids[7:]

    params = build_otm_bank(
        generator,
        pca,
        "otm_full",
        bank_size=len(support),
        seed=13,
    )
    query = rng.normal(size=(11, 6)).astype(np.float32)
    score = calculate_otm_scores(
        query,
        pca,
        params,
        "otm_full",
        bank_chunk=5,
    )

    mu = np.asarray(pca["mu"], dtype=np.float32)
    components = np.asarray(pca["components"], dtype=np.float32)
    projected_bank = mu + params["bank"] @ components.T
    direct = np.min(
        np.sum((query[:, None, :] - projected_bank[None, :, :]) ** 2, axis=2),
        axis=1,
    )
    np.testing.assert_allclose(score, direct, rtol=2e-5, atol=2e-5)


def test_in_subspace_blind_spot_and_self_exclusion():
    pca = {
        "mu": np.zeros(2, dtype=np.float64),
        "components": np.array([[1.0], [0.0]], dtype=np.float64),
        "eigvals": np.ones(1, dtype=np.float64),
        "k": 1,
        "eps": 1e-6,
    }
    support = np.array([[-3.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    source_ids = np.array([0, 1], dtype=np.int64)

    def generator():
        yield support, source_ids

    tangent = build_otm_bank(
        generator, pca, "tangent", bank_size=2, seed=2
    )
    query = np.array([[0.0, 0.0]], dtype=np.float32)
    q_score = calculate_otm_scores(query, pca, None, "baseline")
    t_score = calculate_otm_scores(query, pca, tangent, "tangent")
    np.testing.assert_allclose(q_score, [0.0], atol=1e-7)
    np.testing.assert_allclose(t_score, [9.0], atol=1e-6)

    support_query = np.array([[-3.0, 0.0]], dtype=np.float32)
    excluded = calculate_otm_scores(
        support_query,
        pca,
        tangent,
        "tangent",
        query_source_ids=np.array([0], dtype=np.int64),
        bank_chunk=1,
    )
    np.testing.assert_allclose(excluded, [36.0], atol=1e-6)


def test_streaming_bank_is_deterministic_and_bounded():
    rng = np.random.default_rng(17)
    pca = make_pca(rng, feature_dim=5, retained_dim=2)
    support = rng.normal(size=(100, 5)).astype(np.float32)
    ids = np.repeat(np.arange(10), 10).astype(np.int64)

    def generator():
        for start in range(0, 100, 11):
            yield support[start : start + 11], ids[start : start + 11]

    first = build_otm_bank(generator, pca, "tangent", bank_size=23, seed=9)
    second = build_otm_bank(generator, pca, "tangent", bank_size=23, seed=9)
    assert first["bank"].shape == (23, 2)
    np.testing.assert_array_equal(first["bank"], second["bank"])
    np.testing.assert_array_equal(first["raw_support"], second["raw_support"])
    np.testing.assert_array_equal(first["source_ids"], second["source_ids"])


def test_decoupling_and_cota_guards():
    rng = np.random.default_rng(29)
    pca = make_pca(rng, feature_dim=8, retained_dim=4)
    support = rng.normal(size=(96, 8)).astype(np.float32)
    ids = np.repeat(np.arange(8), 12).astype(np.int64)

    def generator():
        for start in range(0, len(support), 17):
            yield support[start : start + 17], ids[start : start + 17]

    params = build_otm_bank(
        generator,
        pca,
        "cota",
        bank_size=64,
        seed=7,
    )
    fit_cota_calibration(params, pca, bank_chunk=13)
    calibration = params["cota_calibration"]
    assert len(calibration["indices_a"]) > 0
    assert len(calibration["indices_b"]) > 0
    assert set(calibration["indices_a"]).isdisjoint(calibration["indices_b"])

    query = rng.normal(size=(37, 8)).astype(np.float32)
    baseline = calculate_otm_scores(query, pca, None, "baseline")
    image_decoupled, pixel_decoupled = calculate_task_scores(
        query, pca, params, "otm_decoupled", bank_chunk=11
    )
    np.testing.assert_array_equal(pixel_decoupled, baseline)
    assert np.all(image_decoupled >= pixel_decoupled - 1e-6)

    _, calibrated_q = calculate_task_scores(
        query, pca, params, "calibrated_q", bank_chunk=11
    )
    _, calibrated_sum = calculate_task_scores(
        query, pca, params, "calibrated_sum", bank_chunk=11
    )
    _, cota = calculate_task_scores(
        query, pca, params, "cota", bank_chunk=11
    )
    _, cota_safe = calculate_task_scores(
        query, pca, params, "cota_safe", bank_chunk=11
    )
    assert np.isfinite(calibrated_sum).all()
    assert np.isfinite(cota_safe).all()
    assert np.all(cota >= calibrated_q - 1e-6)
    assert np.all(cota <= 2.0 * calibrated_q + 1e-6)

    q_bins = np.searchsorted(
        calibration["q_sorted"], baseline, side="right"
    )
    order = np.argsort(q_bins, kind="stable")
    ordered_bins = q_bins[order]
    ordered_safe = cota_safe[order]
    for i in range(1, len(order)):
        if ordered_bins[i] > ordered_bins[i - 1]:
            assert ordered_safe[i] > ordered_safe[i - 1]

    zero_query = params["raw_support"][:5]
    zero_ids = params["source_ids"][:5]
    expected_image, _ = calculate_task_scores(
        zero_query,
        pca,
        params,
        "otm_full",
        query_source_ids=zero_ids,
        bank_chunk=11,
    )
    cota_image, _ = calculate_task_scores(
        zero_query,
        pca,
        params,
        "cota",
        query_source_ids=zero_ids,
        bank_chunk=11,
    )
    np.testing.assert_allclose(cota_image, expected_image, rtol=1e-6, atol=1e-6)


def main():
    test_baseline_is_exact()
    test_pythagorean_identity()
    test_in_subspace_blind_spot_and_self_exclusion()
    test_streaming_bank_is_deterministic_and_bounded()
    test_decoupling_and_cota_guards()
    print(
        "PASS: baseline routing, OTM geometry, batched self-exclusion, "
        "streaming bank, task decoupling, and COTA guards succeeded."
    )


if __name__ == "__main__":
    main()
