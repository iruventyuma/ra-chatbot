"""
Chat-first Streamlit interface for the RA Assistant.

  - Full UI localization: switching the language in the sidebar changes
    every visible label, not just the chatbot's own answers (see
    src/ui_strings.py -- translations are a first pass, not
    professionally reviewed; worth a native-speaker check before
    presenting this to real users).
  - Guided in-chat symptom flow: after an X-ray is analyzed, or via the
    sidebar button, the assistant asks the 5 ACR/EULAR questions one at a
    time IN THE CHAT, using chatbot.py's rule-based parser (reliable,
    deterministic -- not free-text LLM extraction, which could silently
    misread an answer and corrupt the score). The sidebar form is still
    there as a non-chat alternative for the same 5 fields.
  - Gemini also appends ONE organic follow-up question to normal answers
    until symptoms are collected (see generate_grounded_answer's
    symptoms_collected flag), so the conversation feels interactive even
    outside the guided flow.

Run with:  streamlit run app.py
"""
import streamlit as st
from pathlib import Path
from config import SUPPORTED_LANGUAGES, FINETUNED_VIT_DIR
from src.chatbot import RAChatSession
from src.ui_strings import UI_STRINGS

st.set_page_config(page_title="RA Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    color: white;
    margin-bottom: 1.5rem;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }

.dx-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.2rem;
    border-left: 6px solid;
}
.dx-card.positive { background: #fff1f2; border-color: #e11d48; }
.dx-card.negative { background: #f0fdf4; border-color: #16a34a; }
.dx-card h4 { margin: 0 0 0.4rem 0; }
.dx-card .score-badge {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    color: white;
}
.dx-card.positive .score-badge { background: #e11d48; }
.dx-card.negative .score-badge { background: #16a34a; }
.dx-card .disclaimer { font-size: 0.82rem; color: #555; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Session state
# ============================================================
if "language_code" not in st.session_state:
    st.session_state.language_code = "en"
if "session" not in st.session_state:
    st.session_state.session = RAChatSession(language_code=st.session_state.language_code)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "score_result" not in st.session_state:
    st.session_state.score_result = None
if "image_analyzed" not in st.session_state:
    st.session_state.image_analyzed = False

T = UI_STRINGS[st.session_state.language_code]  # shorthand for this render pass


def _push_assistant_message(text: str):
    st.session_state.chat_history.append(("assistant", text))


st.markdown(f"""
<div class="main-header">
  <h1>{T['app_title']}</h1>
  <p>{T['app_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar: language, image upload, symptom intake
# ============================================================
with st.sidebar:
    st.markdown(f"### {T['language_label']}")
    new_lang = st.selectbox(
        T["language_label"],
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda c: SUPPORTED_LANGUAGES[c],
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.language_code),
        label_visibility="collapsed",
    )
    if new_lang != st.session_state.language_code:
        old = st.session_state.session
        st.session_state.language_code = new_lang
        new_session = RAChatSession(language_code=new_lang)
        new_session.image_path = old.image_path
        new_session.similar_cases = old.similar_cases
        new_session.symptom_answers = old.symptom_answers
        new_session.classifier_prediction = old.classifier_prediction
        st.session_state.session = new_session
        st.rerun()

    st.divider()

    st.markdown(f"### {T['xray_header']}")
    uploaded_file = st.file_uploader(
        T["xray_upload_label"], type=["bmp", "jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    if uploaded_file is not None and not st.session_state.image_analyzed:
        temp_path = Path("temp_upload") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        temp_path.write_bytes(uploaded_file.getbuffer())

        with st.spinner(T["spinner_analyzing"]):
            cases = st.session_state.session.upload_image(str(temp_path))
        st.success(T["found_cases"].format(n=len(cases)))

        if FINETUNED_VIT_DIR.exists():
            from src.gradcam import generate_gradcam, predict_with_confidence
            with st.spinner(T["spinner_classifier"]):
                pred = predict_with_confidence(str(temp_path))
                cam_path = str(temp_path).replace(".", "_gradcam.")
                generate_gradcam(str(temp_path), cam_path)
            st.session_state.session.set_classifier_prediction(pred)
            st.info(T["prediction_label"].format(label=pred["predicted_label"], confidence=f"{pred['confidence']:.1%}"))
            st.image(cam_path, caption=T["gradcam_caption"], use_container_width=True)
        else:
            st.caption(T["no_classifier"])

        st.session_state.image_analyzed = True

        # Proactively start the guided symptom flow IN THE CHAT right after
        # an image is analyzed -- "ask questions based on the image".
        if not st.session_state.session.symptom_answers and not st.session_state.session.guided_flow_active():
            first_question = st.session_state.session.start_guided_symptom_flow()
            _push_assistant_message(f"{T['guided_intro']}\n\n{first_question}")

    # Symptom intake happens entirely in the chat now (see below) -- no
    # sidebar form. The guided flow auto-starts as the first chat message.

# ============================================================
# Main: classification summary card (if computed) + chat
# ============================================================
if st.session_state.score_result:
    result = st.session_state.score_result
    css_class = "positive" if result["classified_as_ra"] else "negative"
    threshold_label = T["dx_positive_label"] if result["classified_as_ra"] else T["dx_negative_label"]
    st.markdown(f"""
    <div class="dx-card {css_class}">
        <h4>{'⚠️' if result['classified_as_ra'] else '✅'} {T.get('classification_summary_title', 'Classification summary')}
            <span class="score-badge">{result['total_score']}/{result['max_score']}</span>
        </h4>
        <p><b>{threshold_label}</b> {T['dx_cutoff_note']}</p>
        <div class="disclaimer">{T['disclaimer_text']}</div>
    </div>
    """, unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    avatar = "🧑" if role == "user" else "🩺"
    with st.chat_message(role, avatar=avatar):
        st.write(msg)

if not st.session_state.chat_history:
    # Auto-start the guided symptom flow as the FIRST thing in the chat,
    # instead of a generic welcome message -- per your request, all
    # symptom intake now happens in the chat itself, no sidebar form.
    session = st.session_state.session
    if not session.symptom_answers and not session.guided_flow_active():
        first_question = session.start_guided_symptom_flow()
        _push_assistant_message(f"{T['welcome_message']}\n\n{T['guided_intro']}\n\n{first_question}")
        st.rerun()

def _handle_user_message(text: str):
    """Shared path for both typed chat input and quick-reply button taps."""
    st.session_state.chat_history.append(("user", text))
    session = st.session_state.session
    if session.guided_flow_active():
        # Deterministic rule-based parsing (chatbot.py), not LLM free-text
        # extraction -- a misread answer here would silently corrupt the
        # ACR/EULAR score.
        result = session.submit_guided_answer(text)
        if isinstance(result, dict):
            st.session_state.score_result = result
            reply = result.get("report_text", T["guided_done"])
        else:
            reply = result  # next guided question, or the confirmation summary
    else:
        with st.spinner(T["spinner_answer"]):
            reply = session.answer_question(text)
    st.session_state.chat_history.append(("assistant", reply))


session = st.session_state.session

# Quick-reply buttons for the current guided question -- lets someone tap
# an answer instead of typing/reading medical terms or exact numbers.
# Important for patients who may not be comfortable with either.
guided_options = session.current_guided_options()
if guided_options:
    cols = st.columns(len(guided_options))
    for col, (label, _value) in zip(cols, guided_options):
        if col.button(label, use_container_width=True, key=f"qr_{label}_{len(st.session_state.chat_history)}"):
            _handle_user_message(label)
            st.rerun()

user_msg = st.chat_input(T["chat_placeholder"])
if user_msg:
    _handle_user_message(user_msg)
    st.rerun()