import numpy as np

from rag.budget_features import assemble_features


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
