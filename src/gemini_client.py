"""
Thin wrapper around the Gemini API.
Handles: grounded answer generation, and the translate-in/translate-out
steps for the multilingual layer.
"""

from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL, SUPPORTED_LANGUAGES


# Create Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def translate_to_english(text: str, source_lang_code: str) -> str:
    """Translate a user query into English before it hits the RAG pipeline."""

    if source_lang_code == "en":
        return text

    lang_name = SUPPORTED_LANGUAGES.get(
        source_lang_code,
        source_lang_code
    )

    prompt = (
        f"Translate the following {lang_name} medical text to English. "
        f"Preserve medical meaning exactly. "
        f"Return ONLY the translation, nothing else.\n\n"
        f"Text: {text}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text.strip()


def translate_from_english(text: str, target_lang_code: str) -> str:
    """Translate the generated answer back into the user's language."""

    if target_lang_code == "en":
        return text

    lang_name = SUPPORTED_LANGUAGES.get(
        target_lang_code,
        target_lang_code
    )

    prompt = (
        f"Translate the following English medical explanation to {lang_name}. "
        f"Keep it clear and easy for a patient to understand. "
        f"Return ONLY the translation, nothing else.\n\n"
        f"Text: {text}"
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text.strip()


def generate_grounded_answer(
    user_question: str,
    image_context: str,
    acr_eular_context: str,
    guideline_passages: list[str],
    symptoms_collected: bool = False,
) -> str:
    """
    Combine everything retrieved (image findings, ACR/EULAR score,
    guideline text) into one prompt and ask Gemini to synthesize
    a grounded answer.

    This is the core RAG generation step -- Gemini should use only
    the context provided in the prompt.

    symptoms_collected: whether the user has already submitted the
    structured ACR/EULAR symptom form. If not, the prompt asks Gemini to
    end its answer with ONE targeted follow-up question, so the chat feels
    interactive rather than a single one-shot answer. If symptoms are
    already collected, Gemini is told explicitly not to re-ask for them.
    """

    passages_block = "\n\n".join(
        f"[Source {i + 1}] {p}"
        for i, p in enumerate(guideline_passages)
    )

    if symptoms_collected:
        followup_instruction = (
            "The patient has already provided their symptom details, reflected in the "
            "ACR/EULAR classification below. Do NOT ask them for information that has "
            "already been provided (joint counts, RF/anti-CCP status, CRP/ESR, symptom "
            "duration)."
        )
    else:
        followup_instruction = (
            "The patient has NOT yet provided detailed symptom information (specific "
            "joint counts, RF/anti-CCP blood test result, CRP/ESR result, or symptom "
            "duration). After answering their question as well as you can with the "
            "context available, end your response by asking exactly ONE specific, "
            "relevant follow-up question that would gather the single most useful "
            "missing piece of symptom information relevant to their question -- so the "
            "conversation feels like a real intake interview, not a one-shot answer. "
            "Ask only one question at a time."
        )

    prompt = f"""You are a rheumatoid arthritis (RA) assistant.

Answer using ONLY the context provided below.
Do not add medical facts that aren't grounded in this context.

Explicitly state that this is a classification SUGGESTION, not a diagnosis,
and recommend confirmation by a rheumatologist.

{followup_instruction}

USER QUESTION:
{user_question}

RETRIEVED IMAGE FINDINGS (similar-severity X-ray cases):
{image_context}

CALCULATED ACR/EULAR CLASSIFICATION:
{acr_eular_context}

RETRIEVED GUIDELINE / LITERATURE PASSAGES:
{passages_block}

Write a clear, patient-friendly answer.

Cite which source(s) support each claim using [Source N] notation.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text.strip()