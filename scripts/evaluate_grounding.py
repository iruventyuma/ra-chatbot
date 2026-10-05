"""
Grounding / hallucination evaluation for the RAG pipeline's generated
answers, for your report's evaluation section.

Method (a standard, lightweight approach when you don't have human
annotators): split the generated answer into sentences, embed each
sentence and every retrieved passage with the same sentence-embedding
model already used for retrieval, and take each answer sentence's best
cosine-similarity match against the passages. A sentence with no good
match to any retrieved passage is flagged as "ungrounded" -- it's making
a claim the retrieved context doesn't support, which is the working
definition of a hallucination risk in a RAG system.

This is a heuristic, not a certified metric -- report it as "N% of
answer sentences had a supporting passage above cosine similarity T",
and be upfront in your write-up that it's a proxy, not human-verified
grounding.

Run:
    python scripts/evaluate_grounding.py
"""
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer, util
from config import TEXT_EMBED_MODEL_NAME
from src.chatbot import RAChatSession

GROUNDING_THRESHOLD = 0.45  # cosine similarity below this = flagged as ungrounded

_embed_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)


def _split_sentences(text: str) -> list[str]:
    # Simple sentence splitter -- good enough for this heuristic; swap for
    # a proper sentence tokenizer (e.g. nltk.sent_tokenize) if you want it
    # more robust for the report.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) > 3]  # drop stray fragments


def evaluate_grounding(answer: str, passages: list[str], threshold: float = GROUNDING_THRESHOLD) -> dict:
    sentences = _split_sentences(answer)
    if not sentences:
        return {"num_sentences": 0, "grounded": 0, "ungrounded": 0, "grounding_rate": None, "details": []}
    if not passages:
        return {
            "num_sentences": len(sentences),
            "grounded": 0,
            "ungrounded": len(sentences),
            "grounding_rate": 0.0,
            "details": [{"sentence": s, "best_score": 0.0, "grounded": False} for s in sentences],
        }

    sentence_embs = _embed_model.encode(sentences, convert_to_tensor=True)
    passage_embs = _embed_model.encode(passages, convert_to_tensor=True)
    sims = util.cos_sim(sentence_embs, passage_embs).cpu().numpy()  # [num_sentences, num_passages]

    details = []
    grounded_count = 0
    for i, sentence in enumerate(sentences):
        best_score = float(sims[i].max())
        is_grounded = best_score >= threshold
        grounded_count += int(is_grounded)
        details.append({"sentence": sentence, "best_score": round(best_score, 3), "grounded": is_grounded})

    return {
        "num_sentences": len(sentences),
        "grounded": grounded_count,
        "ungrounded": len(sentences) - grounded_count,
        "grounding_rate": grounded_count / len(sentences),
        "details": details,
    }


# A handful of test questions to run through the live pipeline. Add your
# own -- ideally ones you'd expect a rheumatologist to actually ask.
TEST_QUESTIONS = [
    "What does a high anti-CCP result mean for my diagnosis?",
    "How is bone erosion different from joint space narrowing?",
    "Why does morning stiffness matter for diagnosing RA?",
]


def main():
    session = RAChatSession(language_code="en")
    all_results = []
    for question in TEST_QUESTIONS:
        answer = session.answer_question(question)
        # Re-run the same retrieval the chatbot used, so we evaluate against
        # exactly what it was grounded in.
        from src.text_index import hybrid_search
        passages = [p["text"] for p in hybrid_search(question)]

        result = evaluate_grounding(answer, passages)
        all_results.append(result)

        print(f"\nQ: {question}")
        print(f"A: {answer}")
        if result["grounding_rate"] is None:
            print("  (no sentences to evaluate)")
        else:
            print(f"  Grounding: {result['grounded']}/{result['num_sentences']} sentences "
                  f"({result['grounding_rate']:.0%}) had a supporting passage above "
                  f"similarity {GROUNDING_THRESHOLD}")
            for d in result["details"]:
                if not d["grounded"]:
                    print(f"    UNGROUNDED (score {d['best_score']}): {d['sentence']}")

    rates = [r["grounding_rate"] for r in all_results if r["grounding_rate"] is not None]
    if rates:
        print(f"\nOverall grounding rate across {len(rates)} question(s): {np.mean(rates):.0%}")


if __name__ == "__main__":
    main()
