from rag.budget_policy import (
    evaluate_policy_at_threshold,
    oracle_budget,
    oracle_frontier_point,
    policy_per_example,
    select_budget,
    static_frontier,
)

KEEPS_CHEAP_FIRST = [0.1, 0.2, 0.3, 0.5, 1.0]
KEEPS_DESC = [1.0, 0.5, 0.3, 0.2, 0.1]


def test_oracle_budget_picks_smallest_correct():
    scores = {0.1: 0.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0}
    assert oracle_budget(KEEPS_DESC, scores) == 0.2


def test_oracle_budget_unsolved_is_none():
    scores = {k: 0.0 for k in KEEPS_DESC}
    assert oracle_budget(KEEPS_DESC, scores) is None


def test_oracle_budget_only_full_res():
    scores = {0.1: 0.0, 0.2: 0.0, 0.3: 0.0, 0.5: 0.0, 1.0: 1.0}
    assert oracle_budget(KEEPS_DESC, scores) == 1.0


def test_select_budget_picks_cheapest_above_threshold():
    probs = {0.1: 0.2, 0.2: 0.4, 0.3: 0.9, 0.5: 0.95, 1.0: 0.99}
    assert select_budget(KEEPS_CHEAP_FIRST, probs, threshold=0.5) == 0.3


def test_select_budget_falls_back_to_most_expensive():
    # Nothing clears threshold -> safest (most expensive) budget.
    probs = {k: 0.1 for k in KEEPS_CHEAP_FIRST}
    assert select_budget(KEEPS_CHEAP_FIRST, probs, threshold=0.5) == 1.0


def test_select_budget_threshold_zero_picks_cheapest():
    probs = {k: 0.0 for k in KEEPS_CHEAP_FIRST}
    assert select_budget(KEEPS_CHEAP_FIRST, probs, threshold=0.0) == 0.1


def _ex(scores, tokens, probs):
    return {"scores": scores, "tokens": tokens, "probs": probs}


def test_evaluate_policy_at_threshold_lookup():
    # One example: cheapest passing budget is 0.3 (prob 0.8 >= 0.5).
    ex = _ex(
        scores={0.1: 0.0, 0.2: 0.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
        tokens={0.1: 300, 0.2: 600, 0.3: 900, 0.5: 1500, 1.0: 3000},
        probs={0.1: 0.1, 0.2: 0.2, 0.3: 0.8, 0.5: 0.9, 1.0: 0.99},
    )
    acc, tok = evaluate_policy_at_threshold([ex], KEEPS_CHEAP_FIRST, threshold=0.5)
    assert acc == 1.0 and tok == 900


def test_policy_per_example_preserves_order_and_pairs():
    exs = [
        _ex({k: 1.0 for k in KEEPS_DESC}, {k: 100 * i for i, k in enumerate(KEEPS_CHEAP_FIRST, 1)},
            {0.1: 0.9, 0.2: 0.9, 0.3: 0.9, 0.5: 0.9, 1.0: 0.9}),  # picks 0.1 -> tok 100
        _ex({0.1: 0.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 400, 1.0: 500},
            {0.1: 0.1, 0.2: 0.9, 0.3: 0.9, 0.5: 0.9, 1.0: 0.9}),  # picks 0.2 -> tok 200
    ]
    scores, toks = policy_per_example(exs, KEEPS_CHEAP_FIRST, threshold=0.5)
    assert scores == [1.0, 1.0]
    assert toks == [100, 200]


def test_static_frontier_matches_means():
    exs = [
        _ex({0.1: 1.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 500, 1.0: 1000}, {}),
        _ex({0.1: 0.0, 0.2: 0.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 500, 1.0: 1000}, {}),
    ]
    fr = dict((k, (a, t)) for k, a, t in static_frontier(exs, KEEPS_CHEAP_FIRST))
    assert fr[0.1] == (0.5, 100)   # one right one wrong
    assert fr[0.3] == (1.0, 300)   # both right


def test_oracle_frontier_is_upper_bound():
    # ex1 oracle 0.1 (tok100), ex2 oracle 0.3 (tok300), ex3 unsolved -> full res, wrong.
    exs = [
        _ex({0.1: 1.0, 0.2: 1.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 500, 1.0: 1000}, {}),
        _ex({0.1: 0.0, 0.2: 0.0, 0.3: 1.0, 0.5: 1.0, 1.0: 1.0},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 500, 1.0: 1000}, {}),
        _ex({k: 0.0 for k in KEEPS_DESC},
            {0.1: 100, 0.2: 200, 0.3: 300, 0.5: 500, 1.0: 1000}, {}),
    ]
    acc, tok = oracle_frontier_point(exs, KEEPS_CHEAP_FIRST)
    assert abs(acc - 2 / 3) < 1e-9          # two solved, one unsolved
    assert abs(tok - (100 + 300 + 1000) / 3) < 1e-9
