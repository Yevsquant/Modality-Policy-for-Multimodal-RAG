from typing import Dict, List

def build_prompt(query: str, retrieval: Dict, cached_img: Dict) -> str:
    blocks = []
    blocks.append("Answer the question using only the retrieved evidence.")
    blocks.append("If the evidence is insufficient, say that clearly.")
    blocks.append("Cite quote IDs you used in brackets, e.g. [text3] [image2].")
    blocks.append("")
    blocks.append(f"Question: {query}")
    blocks.append("")

    if retrieval["selected_text_quotes"]:
        blocks.append("Retrieved text evidence:")
        for q in retrieval["selected_text_quotes"]:
            blocks.append(
                f"- [{q['quote_id']}]: {q.get('text','')}"
            )
        blocks.append("")

    if retrieval["selected_img_quotes"] and cached_img is not None:
        blocks.append("Retrieved image evidence:")
        if retrieval["selected_img_quotes"]:
            for q in retrieval["selected_img_quotes"]:
                blocks.append(
                    f"- [{q['quote_id']}]"
                )
        if cached_img is not None:
            for q in cached_img:
                blocks.append(
                    f"- [{q['quote_id']}]"
                )
        blocks.append("")

    blocks.append("Return a short answer.")
    return "\n".join(blocks)