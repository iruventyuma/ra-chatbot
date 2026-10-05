"""
Multilingual consistency evaluation for the En/Hi/Te translate-wrapper
layer.

Method:
1. Ask the same English question through the chatbot.
2. Generate the English grounded answer once.
3. Translate that answer into Hindi and Telugu.
4. Translate the Hindi and Telugu answers back to English.
5. Compute pairwise semantic similarity between the three
   English-equivalent answers using sentence embeddings.

This measures multilingual consistency, not medical correctness.
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer, util

from config import TEXT_EMBED_MODEL_NAME
from src.gemini_client import (
    generate_grounded_answer,
    translate_from_english,
    translate_to_english,
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

_embed_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    "What does a high anti-CCP result mean for my diagnosis?",
    "How is bone erosion different from joint space narrowing?",
]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_consistency_for_question(question_en: str) -> dict:

    answers_native = {}
    answers_en = {}

    # --------------------------------------------------------
    # STEP 1: Generate ONE English grounded answer
    # --------------------------------------------------------

    print("\n  Generating English grounded answer...")

    english_answer = generate_grounded_answer(
        user_question=question_en,
        image_context=[],
        acr_eular_context=[],
        guideline_passages=[]
    )

    answers_native["en"] = english_answer
    answers_en["en"] = english_answer

    # --------------------------------------------------------
    # STEP 2: Translate English answer into Hindi and Telugu
    # --------------------------------------------------------

    for lang_code in ["hi", "te"]:

        print(f"\n  Translating answer to: {lang_code}")

        answer_native = translate_from_english(
            english_answer,
            lang_code
        )

        answers_native[lang_code] = answer_native

        # ----------------------------------------------------
        # STEP 3: Translate Hindi/Telugu back to English
        # ----------------------------------------------------

        print(f"  Back-translating {lang_code} → English...")

        answer_en = translate_to_english(
            answer_native,
            lang_code
        )

        answers_en[lang_code] = answer_en

        # ----------------------------------------------------
        # RATE LIMIT WAIT
        # ----------------------------------------------------

        if lang_code != "te":
            print("  Waiting 65 seconds for Gemini rate limit...")
            time.sleep(65)

    # ========================================================
    # SEMANTIC SIMILARITY
    # ========================================================

    codes = ["en", "hi", "te"]

    embeddings = _embed_model.encode(
        [
            answers_en[c]
            for c in codes
        ],
        convert_to_tensor=True
    )

    sim_matrix = util.cos_sim(
        embeddings,
        embeddings
    ).cpu().numpy()

    pairwise = {}

    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):

            pairwise[f"{codes[i]}-{codes[j]}"] = float(
                sim_matrix[i][j]
            )

    return {
        "answers": {
            lang: {
                "native": answers_native[lang],
                "back_translated_en": answers_en[lang]
            }
            for lang in codes
        },
        "pairwise_similarity": pairwise
    }


# ============================================================
# MAIN
# ============================================================

def main():

    all_scores = []

    for question in TEST_QUESTIONS:

        print("\n" + "=" * 70)
        print(f"Q: {question}")
        print("=" * 70)

        try:

            result = evaluate_consistency_for_question(
                question
            )

            print("\n  Pairwise semantic similarity:")

            for pair, score in result["pairwise_similarity"].items():

                print(
                    f"    {pair}: similarity = {score:.3f}"
                )

                all_scores.append(score)

        except Exception as e:

            print("\n  Evaluation failed:")
            print(f"  {type(e).__name__}: {e}")

            print(
                "\n  Skipping this question and continuing..."
            )

            continue

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if all_scores:

        print("\n" + "=" * 70)

        print(
            f"Average cross-language consistency across "
            f"{len(TEST_QUESTIONS)} question(s): "
            f"{np.mean(all_scores):.3f}"
        )

        print("=" * 70)

        print(
            "\nThis is a consistency proxy, not a correctness "
            "check. High similarity means the answers preserve "
            "similar meaning across languages."
        )

        print(
            "Low similarity indicates that the corresponding "
            "language outputs should be manually inspected."
        )

    else:

        print(
            "\nNo successful multilingual evaluation results "
            "were produced."
        )


if __name__ == "__main__":
    main()