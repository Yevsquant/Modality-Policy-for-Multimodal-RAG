import numpy as np

from rag.budget_features import (
    SPATIAL_FEATURE_NAMES,
    assemble_features,
    spatial_features_from_map,
)


def test_assemble_shape_and_cosine():
    img = np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32")
    txt = np.array([[1.0, 0.0], [1.0, 0.0]], dtype="float32")
    feat = assemble_features(img, txt)
    assert feat.shape == (2, 5)  # 2 + 2 + 1
    # row 0: img==txt -> cos 1.0 ; row 1: orthogonal -> cos 0.0
    assert abs(feat[0, -1] - 1.0) < 1e-6
    assert abs(feat[1, -1] - 0.0) < 1e-6
    # first 2 cols are the image emb, next 2 the text emb
    assert np.allclose(feat[:, :2], img)
    assert np.allclose(feat[:, 2:4], txt)


def test_spatial_features_peaky_vs_diffuse():
    grid = (7, 7)
    # A sharply peaked map: one hot patch at the corner, everything else ~0.
    peaky = np.zeros(49, dtype="float32")
    peaky[0] = 1.0  # top-left corner patch
    fp = spatial_features_from_map(peaky, grid)
    names = SPATIAL_FEATURE_NAMES[: len(fp)]
    f = dict(zip(names, fp))

    # A flat/diffuse map: every patch equal.
    flat = np.full(49, 0.3, dtype="float32")
    fd = dict(zip(names, spatial_features_from_map(flat, grid)))

    # Peaky map concentrates mass -> higher top1, lower entropy than diffuse.
    assert f["rel_top1_mass"] > fd["rel_top1_mass"]
    assert f["rel_entropy"] < fd["rel_entropy"]
    # Peak sits at a corner -> far from center; diffuse map's argmax is arbitrary
    # but the corner peak must be clearly off-center.
    assert f["peak_dist_center"] > 0.9
    # Flat map has ~uniform entropy ~1.0
    assert fd["rel_entropy"] > 0.99


def test_spatial_features_finite_and_sized():
    rng = np.random.default_rng(0)
    rel = rng.normal(size=49).astype("float32")
    f = spatial_features_from_map(rel, (7, 7))
    assert f.shape == (9,)  # the 9 map-derived features
    assert np.all(np.isfinite(f))
