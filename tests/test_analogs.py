from pathlib import Path

import numpy as np
import pytest

from adoptsim.data import make_synthetic_dataset
from adoptsim.embeddings import CachedEncoder, Encoder, TfidfEncoder
from adoptsim.evaluation import (
    bootstrap_ci,
    cosine_distance_matrix,
    knn_loo_predict,
    mantel_test,
    mean_loo_predict,
    paired_permutation_test,
    random_analog_loo_error,
)


def _clustered(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 4
    centres = rng.normal(size=(4, 8))
    emb = centres[labels] + 0.1 * rng.normal(size=(n, 8))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    y = labels.astype(float) + 0.05 * rng.normal(size=n)
    return emb, y


def test_knn_beats_mean_on_clustered_data() -> None:
    emb, y = _clustered()
    err_knn = np.abs(knn_loo_predict(emb, y, k=3)[:, 0] - y)
    err_mean = np.abs(mean_loo_predict(y)[:, 0] - y)
    assert err_knn.mean() < 0.5 * err_mean.mean()
    assert paired_permutation_test(err_knn, err_mean, n_perm=999).p_value < 0.01


def test_knn_excludes_self_and_respects_allowed() -> None:
    emb, y = _clustered(n=8)
    allowed = np.tril(np.ones((8, 8), dtype=bool), k=-1)  # only earlier items
    pred = knn_loo_predict(emb, y, k=2, allowed=allowed)
    assert np.isnan(pred[0, 0])
    assert pred[1, 0] == pytest.approx(y[0])


def test_mantel_detects_signal_and_not_noise() -> None:
    emb, y = _clustered()
    d_emb = cosine_distance_matrix(emb)
    d_y = np.abs(y[:, None] - y[None, :])
    assert mantel_test(d_emb, d_y, n_perm=499).p_value < 0.01
    rng = np.random.default_rng(1)
    d_noise = np.abs(rng.normal(size=40)[:, None] - rng.normal(size=40)[None, :])
    d_noise = (d_noise + d_noise.T) / 2
    assert mantel_test(d_emb, d_noise, n_perm=499).p_value > 0.01


def test_random_analog_error_shape() -> None:
    _, y = _clustered(n=10)
    err = random_analog_loo_error(y, k=3, n_rep=20)
    assert err.shape == (10, 1) and np.all(err >= 0)


def test_bootstrap_ci_contains_mean() -> None:
    x = np.random.default_rng(0).normal(1.0, 1.0, 200)
    lo, hi = bootstrap_ci(x, n_boot=2000)
    assert lo < x.mean() < hi


def test_tfidf_positive_control_on_synthetic() -> None:
    _, descs, truth = make_synthetic_dataset(n=40, signal=0.9, seed=0)
    emb = TfidfEncoder().encode(descs["description"].tolist())
    lq = np.log(truth["q"].to_numpy())
    d_y = np.abs(lq[:, None] - lq[None, :])
    assert mantel_test(cosine_distance_matrix(emb), d_y, n_perm=499).p_value < 0.01


class _Counting(Encoder):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "counting/test"

    def _encode(self, texts):  # type: ignore[no-untyped-def]
        self.calls += len(texts)
        return np.array([[len(t), 1.0] for t in texts])


def test_cache_reuses_vectors(tmp_path: Path) -> None:
    inner = _Counting()
    enc = CachedEncoder(inner, cache_dir=tmp_path)
    a = enc.encode(["ab", "abc"])
    b = enc.encode(["abc", "abcd"])
    assert inner.calls == 3
    np.testing.assert_allclose(a[1], b[0])


def test_cache_refuses_corpus_dependent_encoder(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CachedEncoder(TfidfEncoder(), cache_dir=tmp_path)
