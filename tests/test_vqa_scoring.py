from rag.vqa_scoring import anls, extract_choice, mc_score


def test_extract_choice_forms():
    assert extract_choice("(A)") == "A"
    assert extract_choice("A.") == "A"
    assert extract_choice("B") == "B"
    assert extract_choice("The answer is C") == "C"
    assert extract_choice("answer: d", valid="ABCD") == "D"
    assert extract_choice("none of these letters", valid="ABCD") is None
    assert extract_choice("") is None


def test_mc_score():
    assert mc_score("(A) rubber", "A") == 1.0
    assert mc_score("B", "A") == 0.0
    assert mc_score("The answer is D.", "D") == 1.0


def test_anls_perfect_and_zero():
    assert anls("hello world", ["hello world"]) == 1.0
    assert anls("xyz", ["completely different"]) == 0.0


def test_anls_near_match_high_and_best_over_golds():
    # one char off in a 7-char string -> sim ~0.857, above tau
    s = anls("morning", ["mornong"])
    assert 0.8 < s < 1.0
    # best over multiple golds
    assert anls("cat", ["dog", "cat"]) == 1.0


def test_anls_below_tau_zeroed():
    # half wrong -> below 0.5 threshold -> 0
    assert anls("ab", ["zy"]) == 0.0
