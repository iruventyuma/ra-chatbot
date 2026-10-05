"""
Orchestrates the full pipeline, matching the flow we designed:

  image upload -> retrieve similar scored cases (ViT + FAISS)
  -> ask ACR/EULAR-based symptom questions
  -> compute ACR/EULAR score
  -> hybrid retrieve guideline text (FAISS + BM25)
  -> Gemini generates the grounded answer
  -> (if needed) translate in/out for Hindi/Telugu

This is a simple state machine. Wire it up to app.py (Streamlit) or any
other interface.
"""
from src.image_index import search_similar_cases
from src.text_index import hybrid_search
from src.acr_eular import ACREularInput, calculate_acr_eular
from src.gemini_client import (
    generate_grounded_answer,
    translate_to_english,
    translate_from_english,
)


class RAChatSession:
    """Holds state for one conversation: uploaded image, collected symptom
    answers, and the user's chosen language."""

    def __init__(self, language_code: str = "en"):
        self.language_code = language_code
        self.image_path: str | None = None
        self.similar_cases: list[dict] = []
        self.symptom_answers: dict = {}
        self.classifier_prediction: dict | None = None  # set by app.py if a fine-tuned checkpoint exists
        self._guided_field_index: int | None = None
        self._awaiting_confirmation: bool = False

    # --- Step 1: image intake ---
    def upload_image(self, image_path: str) -> list[dict]:
        self.image_path = image_path
        self.similar_cases = search_similar_cases(image_path)
        return self.similar_cases

    def set_classifier_prediction(self, prediction: dict) -> None:
        """Called from app.py with the fine-tuned ViT's prediction (if a
        checkpoint exists), so it can be grounded into the chat answer
        instead of only being shown once in the upload UI."""
        self.classifier_prediction = prediction

    # --- Step 2: symptom questions (ACR/EULAR-aligned), in plain patient
    # language, each with quick-reply button options so someone doesn't
    # need to know medical terms or type precise numbers/wording. Free
    # text still works as a fallback for whoever prefers to type.
    #
    # Joint-count buttons map to a REPRESENTATIVE count for that bucket
    # (e.g. "A few (1-3)" -> 2), not an exact patient-reported number --
    # this is a deliberate approximation traded for usability, since most
    # patients can't reliably self-count precise joint totals anyway.
    SYMPTOM_QUESTIONS = [
        (
            "small_joints_involved",
            "Let's start simple. Are any of your finger or wrist joints swollen or painful? About how many?",
            [("None", 0), ("A few (1-3)", 2), ("Several (4-10)", 6), ("Many (10+)", 12), ("Not sure", "unknown")],
        ),
        (
            "large_joints_involved",
            "What about bigger joints -- shoulders, elbows, knees, or hips? Any swelling or pain there?",
            [("None", 0), ("One joint", 1), ("A few (2-10)", 4), ("Not sure", "unknown")],
        ),
        (
            "rf_or_accp_positive",
            "Have you ever had a blood test for rheumatoid arthritis (doctors sometimes call it 'RF' or 'anti-CCP')? If you know the result, was it positive or negative?",
            [("Never had this test / not sure", "unknown"), ("Negative / normal", "negative"), ("Positive", "high_positive")],
        ),
        (
            "crp_or_esr_abnormal",
            "Have you had a blood test for inflammation (doctors sometimes call it 'CRP' or 'ESR')? Was it high or abnormal?",
            [("Never had this test / not sure", "unknown"), ("No, it was normal", "no"), ("Yes, high/abnormal", "yes")],
        ),
        (
            "symptom_duration_weeks",
            "Last one -- how long have you had these joint symptoms?",
            [("Less than a week", 0), ("A few weeks", 3), ("More than 6 weeks", 8), ("Not sure", "unknown")],
        ),
    ]

    def record_symptom_answer(self, key: str, value) -> None:
        self.symptom_answers[key] = value

    # --- Guided in-chat symptom flow (structured Q&A, not free-text extraction) ---
    def start_guided_symptom_flow(self) -> str:
        self._guided_field_index = 0
        self._awaiting_confirmation = False
        return self.SYMPTOM_QUESTIONS[0][1]

    def guided_flow_active(self) -> bool:
        return getattr(self, "_guided_field_index", None) is not None or getattr(self, "_awaiting_confirmation", False)

    def current_guided_options(self):
        """
        Quick-reply button options (label, value) for the CURRENT step --
        either the current question, or the yes/no confirmation step at
        the end -- or None if the flow isn't active.
        """
        if getattr(self, "_awaiting_confirmation", False):
            return [("Yes, that's correct", "confirm_yes"), ("No, let me redo it", "confirm_no")]
        if self._guided_field_index is None:
            return None
        return self.SYMPTOM_QUESTIONS[self._guided_field_index][2]

    def _build_confirmation_summary(self) -> str:
        a = self.symptom_answers

        def fmt_joints(v):
            return "not sure" if v == "unknown" else str(v)

        def fmt_rf(v):
            return {"unknown": "not tested / not sure", "negative": "negative",
                    "low_positive": "positive (low)", "high_positive": "positive (high)"}.get(v, str(v))

        def fmt_crp(v):
            return {"unknown": "not tested / not sure", "yes": "abnormal / high", "no": "normal"}.get(v, str(v))

        def fmt_weeks(v):
            return "not sure" if v == "unknown" else f"{v} week(s)"

        lines = [
            "Before I finish, let's make sure I got this right:",
            f"- Small joints (fingers/wrists) affected: {fmt_joints(a.get('small_joints_involved'))}",
            f"- Large joints (shoulders/elbows/knees/hips) affected: {fmt_joints(a.get('large_joints_involved'))}",
            f"- RF/anti-CCP blood test: {fmt_rf(a.get('rf_or_accp_positive'))}",
            f"- CRP/ESR blood test: {fmt_crp(a.get('crp_or_esr_abnormal'))}",
            f"- Symptom duration: {fmt_weeks(a.get('symptom_duration_weeks'))}",
            "",
            "Does this look correct?",
        ]
        return "\n".join(lines)

    def build_final_report_text(self) -> str:
        """
        The final report shown once the patient confirms their answers.
        Deliberately keeps two different kinds of numbers separate and
        clearly labeled: the ACR/EULAR total is a rule-based clinical
        score (not a probability -- it has no "confidence"), while the
        image classifier's percentage (if an X-ray was uploaded and a
        fine-tuned checkpoint exists) IS a real model confidence. Blending
        these into one invented "confidence %" would misrepresent what
        the ACR/EULAR score actually is.
        """
        result = self.compute_score()
        lines = [
            "**Classification report**",
            "",
            "ACR/EULAR score breakdown:",
            f"- Joint involvement: {result['joint_involvement_score']}/5",
            f"- Serology (RF/anti-CCP): {result['serology_score']}/3",
            f"- Acute phase (CRP/ESR): {result['acute_phase_score']}/1",
            f"- Symptom duration: {result['duration_score']}/1",
            f"- **Total score: {result['total_score']}/{result['max_score']}**",
            "",
            f"Classification suggestion: {'Meets' if result['classified_as_ra'] else 'Does not meet'} "
            f"the ACR/EULAR threshold for RA (cutoff: 6/10).",
        ]
        if result.get("has_unknown_fields"):
            lines.append(
                "Note: one or more blood test results were marked as not known / not tested and "
                "were scored as 0 for now -- the real score could be higher if those come back positive."
            )
        if self.classifier_prediction:
            lines += [
                "",
                f"X-ray classifier prediction: {self.classifier_prediction['predicted_label']} "
                f"({self.classifier_prediction['confidence']:.1%} confidence)",
                "(This confidence percentage comes from the image classifier model only. The "
                "ACR/EULAR score above is a rule-based clinical score, not a probability, so it "
                "does not have its own confidence value.)",
            ]
        lines += [
            "",
            "This is a classification suggestion, not a medical diagnosis. "
            "Please confirm any result with a rheumatologist.",
        ]
        return "\n".join(lines)

    def submit_guided_answer(self, raw_text: str):
        """
        Parse the reply for the CURRENT guided-flow question only (one
        known field/type at a time). Accepts either an exact button label
        or free text as a fallback.

        Uncertainty ("don't know", "not sure", never tested, etc.) is
        recorded as the literal string "unknown" -- it is NEVER silently
        defaulted to a definite negative/no. Treating an untested patient
        as a confirmed-negative one would misrepresent their actual
        situation in the ACR/EULAR score and its summary.
        """
        if getattr(self, "_awaiting_confirmation", False):
            raw = raw_text.strip().lower()
            wants_redo = raw in ("no", "confirm_no", "n") or "redo" in raw or "wrong" in raw or "no," in raw
            if wants_redo:
                self.symptom_answers = {}
                self._awaiting_confirmation = False
                self._guided_field_index = 0
                return "No problem, let's go through it again.\n\n" + self.SYMPTOM_QUESTIONS[0][1]
            else:
                self._awaiting_confirmation = False
                result = self.compute_score()
                result["report_text"] = self.build_final_report_text()
                return result

        idx = self._guided_field_index
        key, _, options = self.SYMPTOM_QUESTIONS[idx]
        raw = raw_text.strip().lower()

        matched_value = None
        for label, val in options:
            if raw == label.strip().lower():
                matched_value = val
                break

        uncertain_phrases = ("don't know", "dont know", "not sure", "no idea", "unknown",
                              "never had", "haven't had", "havent had", "no test", "not tested",
                              "what is it", "don't understand", "dont understand")

        if matched_value is not None:
            value = matched_value
        elif any(p in raw for p in uncertain_phrases):
            value = "unknown"
        elif key in ("small_joints_involved", "large_joints_involved"):
            digits = "".join(c for c in raw if c.isdigit())
            if digits:
                value = int(digits)
            elif "none" in raw or raw in ("no", "n"):
                value = 0
            elif "few" in raw:
                value = 2
            elif "several" in raw or "many" in raw:
                value = 6
            else:
                value = "unknown"
        elif key == "rf_or_accp_positive":
            if "high" in raw or ("positive" in raw and "negative" not in raw):
                value = "high_positive"
            elif "low" in raw:
                value = "low_positive"
            elif "negative" in raw or "normal" in raw:
                value = "negative"
            else:
                value = "unknown"
        elif key == "crp_or_esr_abnormal":
            if raw.startswith("y") or "abnormal" in raw or "high" in raw:
                value = "yes"
            elif raw.startswith("n") or "normal" in raw:
                value = "no"
            else:
                value = "unknown"
        elif key == "symptom_duration_weeks":
            digits = "".join(c for c in raw if c.isdigit() or c == ".")
            if digits:
                value = float(digits)
            elif "month" in raw:
                value = 8.0
            elif "week" in raw:
                value = 3.0
            elif "day" in raw or "just" in raw or "start" in raw:
                value = 0.0
            else:
                value = "unknown"
        else:
            value = raw

        self.record_symptom_answer(key, value)
        idx += 1
        if idx < len(self.SYMPTOM_QUESTIONS):
            self._guided_field_index = idx
            return self.SYMPTOM_QUESTIONS[idx][1]
        else:
            self._guided_field_index = None
            self._awaiting_confirmation = True
            return self._build_confirmation_summary()

    # --- Step 3: score ---
    def compute_score(self) -> dict:
        answers = self.symptom_answers

        def _numeric_or_default(key, default):
            v = answers.get(key, default)
            return default if v == "unknown" else v

        acr_input = ACREularInput(
            small_joints_involved=int(_numeric_or_default("small_joints_involved", 0)),
            large_joints_involved=int(_numeric_or_default("large_joints_involved", 0)),
            rf_or_accp_positive=(
                "negative" if answers.get("rf_or_accp_positive") in (None, "unknown")
                else answers["rf_or_accp_positive"]
            ),
            crp_or_esr_abnormal=(
                False if answers.get("crp_or_esr_abnormal") in (None, "unknown")
                else str(answers["crp_or_esr_abnormal"]).lower() == "yes"
            ),
            symptom_duration_weeks=float(_numeric_or_default("symptom_duration_weeks", 0)),
        )
        result = calculate_acr_eular(acr_input)

        unknown_fields = [k for k, v in answers.items() if v == "unknown"]
        result["has_unknown_fields"] = bool(unknown_fields)
        if unknown_fields:
            result["summary"] += (
                f" Note: {len(unknown_fields)} item(s) were marked as not known / not "
                f"tested and were scored as 0 for now -- the real score could be higher "
                f"if those tests come back positive or abnormal. Consider getting these "
                f"tests done for a more complete picture."
            )
        return result

    # --- Step 4 + 5: retrieve guideline text + generate grounded answer ---
    def answer_question(self, user_question: str) -> str:
        # translate in
        question_en = translate_to_english(user_question, self.language_code)

        # build image context string from retrieved similar cases
        image_context_lines = []
        for case in self.similar_cases:
            image_context_lines.append(
                f"- Similar case (similarity {case['similarity']:.2f}): BE/JSN scores = {case['scores']}"
            )
        if self.classifier_prediction:
            image_context_lines.append(
                f"- Fine-tuned classifier prediction on the uploaded X-ray: "
                f"{self.classifier_prediction['predicted_label']} "
                f"(confidence {self.classifier_prediction['confidence']:.1%})"
            )
        image_context = "\n".join(image_context_lines) or "No image uploaded / no similar cases retrieved."

        # score
        score_result = self.compute_score()
        acr_eular_context = score_result["summary"]

        # hybrid text retrieval
        passages = hybrid_search(question_en)
        passage_texts = [p["text"] for p in passages]

        # generate grounded answer
        answer_en = generate_grounded_answer(
            user_question=question_en,
            image_context=image_context,
            acr_eular_context=acr_eular_context,
            guideline_passages=passage_texts,
            symptoms_collected=bool(self.symptom_answers),
        )

        # translate out
        return translate_from_english(answer_en, self.language_code)