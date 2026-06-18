from rag.metrics import bootstrap_ci, paired_diff_ci


def test_bootstrap_ci_constant():
    mean, lo, hi = bootstrap_ci([0.5] * 20, n_boot=200, seed=1)
    assert mean == 0.5
    assert lo == 0.5 and hi == 0.5


def test_bootstrap_ci_brackets_mean():
    vals = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  # mean 0.6
    mean, lo, hi = bootstrap_ci(vals, n_boot=2000, seed=1)
    assert abs(mean - 0.6) < 1e-9
    assert lo <= mean <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (None, None, None)


def test_paired_diff_ci_identical_is_zero():
    a = [1, 0, 1, 1, 0]
    mean, lo, hi = paired_diff_ci(a, a, n_boot=200, seed=1)
    assert mean == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_diff_ci_all_better():
    a = [1, 1, 1, 1]
    b = [0, 0, 0, 0]
    mean, lo, hi = paired_diff_ci(a, b, n_boot=200, seed=1)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0
