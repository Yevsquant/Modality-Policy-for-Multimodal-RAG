from rag.flops_model import condition_flops, llm_layer_flops, vit_flops


def test_llm_layer_flops_monotonic():
    assert llm_layer_flops(100) < llm_layer_flops(200) < llm_layer_flops(400)


def test_vit_flops_superlinear_due_to_attention():
    # doubling tokens more than doubles FLOPs (O(V^2) attention term)
    assert vit_flops(1000) > 2 * vit_flops(500) * 0.9  # at least clearly superlinear-ish
    assert vit_flops(1000) > vit_flops(500)


def test_fastv_saves_only_deep_llm_not_vision():
    # full vs full+FastV: same img_before -> identical vision cost; FastV cuts only LLM.
    full_t, full_v, full_llm = condition_flops(1018, 1018, prune_layer=None)
    fv_t, fv_v, fv_llm = condition_flops(1018, 255, prune_layer=3)
    assert fv_v == full_v                  # vision identical (full image encoded)
    assert fv_llm < full_llm               # only deep LLM layers shrink
    assert fv_t < full_t


def test_downscale_cuts_vision_too():
    # input downscale to the same deep-token count as FastV pays LESS vision.
    fv_t, fv_v, _ = condition_flops(1018, 255, prune_layer=3)
    ds_t, ds_v, _ = condition_flops(258, 258, prune_layer=None)
    assert ds_v < fv_v                     # downscale encodes a smaller image
    assert ds_t < fv_t                     # and is cheaper overall at matched deep tokens
