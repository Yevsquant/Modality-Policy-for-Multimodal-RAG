from typing import Dict, List, Sequence, Union

CachedImg = Union[Sequence[Dict], Dict, None]


def _cached_quotes_list(cached_img: CachedImg) -> List[Dict]:
    if cached_img is None:
        return []
    if isinstance(cached_img, list):
        return list(cached_img)
    return []


def _is_single_image_vqa(retrieval: Dict, cached_img: CachedImg) -> bool:
    texts = retrieval.get("selected_text_quotes") or []
    imgs = retrieval.get("selected_img_quotes") or []
    cached = _cached_quotes_list(cached_img)
    return not texts and len(imgs) + len(cached) == 1


def build_prompt(query: str, retrieval: Dict, cached_img: CachedImg) -> str:
    blocks: List[str] = []
    vqa_mode = _is_single_image_vqa(retrieval, cached_img)

    if vqa_mode:
        blocks.append("Answer the visual question using the image provided below.")
        blocks.append("Reply with a short, direct answer (phrase or few words) when possible.")
    else:
        blocks.append("Answer the question using only the retrieved evidence.")
        blocks.append("If the evidence is insufficient, say that clearly.")
        blocks.append("Cite quote IDs you used in brackets, e.g. [text3] [image2].")

    blocks.append("")
    blocks.append(f"Question: {query}")
    blocks.append("")

    if retrieval.get("selected_text_quotes"):
        blocks.append("Retrieved text evidence:")
        for q in retrieval["selected_text_quotes"]:
            blocks.append(f"- [{q['quote_id']}]: {q.get('text', '')}")
        blocks.append("")

    imgs = retrieval.get("selected_img_quotes") or []
    cached = _cached_quotes_list(cached_img)
    if imgs or cached:
        if vqa_mode:
            blocks.append("Image:")
        else:
            blocks.append("Retrieved image evidence:")
        for q in imgs:
            blocks.append(f"- [{q['quote_id']}]")
        for q in cached:
            blocks.append(f"- [{q['quote_id']}]")
        blocks.append("")

    blocks.append("Return a short answer.")
    return "\n".join(blocks)
