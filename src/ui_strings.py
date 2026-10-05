"""
Static UI text for the Streamlit interface, per language. This is
interface chrome (labels, buttons, headers) -- separate from the RAG
answers themselves, which are already translated by src/gemini_client.py.

Translations are a first pass, not clinically or professionally reviewed --
worth a native-speaker check (Telugu especially, given the project's base)
before relying on the exact wording in front of real users.
"""

UI_STRINGS = {
    "en": {
        "app_title": "🩺 Rheumatoid Arthritis RAG Assistant",
        "app_subtitle": "Ask questions, upload an X-ray, and answer a few symptom questions for a grounded classification suggestion.",
        "language_label": "Language",
        "xray_header": "📷 X-ray (optional)",
        "xray_upload_label": "Upload a hand/wrist X-ray",
        "found_cases": "Found {n} similar reference case(s).",
        "no_classifier": "No fine-tuned classifier found -- run scripts/finetune_vit.py to enable this.",
        "symptom_header": "📋 Symptom check (ACR/EULAR)",
        "symptom_caption": "Answer these for a grounded classification score, or use the guided chat button below.",
        "symptom_small": "Small joints (fingers/wrists) swollen/painful",
        "symptom_large": "Large joints (shoulders/elbows/knees/hips) swollen/painful",
        "symptom_rf": "RF / anti-CCP blood test result",
        "symptom_crp": "CRP/ESR abnormal?",
        "symptom_weeks": "Symptom duration (weeks)",
        "submit_button": "Submit symptoms",
        "guided_chat_button": "🗣️ Ask me these questions in chat instead",
        "chat_placeholder": "Ask about your condition, symptoms, or an uploaded X-ray...",
        "welcome_message": (
            "Hi! I'm your RA assistant. You can ask me anything about rheumatoid "
            "arthritis, upload an X-ray from the sidebar, or fill in the symptom "
            "check for a grounded classification suggestion. What would you like to know?"
        ),
        "dx_positive_label": "Meets ACR/EULAR threshold",
        "dx_negative_label": "Below ACR/EULAR threshold",
        "dx_cutoff_note": "for RA classification (cutoff: 6/10).",
        "disclaimer_text": (
            "This is a classification suggestion based on the ACR/EULAR scoring "
            "rules, not a medical diagnosis. Please confirm any result with a rheumatologist."
        ),
        "voice_input_label": "🎤 Or record your question",
        "voice_output_toggle": "🔊 Read answers aloud",
        "transcribing": "Transcribing your voice message...",
        "rf_options": {"negative": "Negative", "low_positive": "Low positive", "high_positive": "High positive"},
        "crp_options": {"no": "No", "yes": "Yes"},
        "prediction_label": "Prediction: **{label}** ({confidence} confidence)",
        "gradcam_caption": "Grad-CAM: where the model focused",
        "spinner_analyzing": "Analyzing image and retrieving similar cases...",
        "spinner_classifier": "Running classifier and generating Grad-CAM...",
        "spinner_answer": "Retrieving and generating a grounded answer...",
        "guided_intro": "Sure -- let's go through a few quick symptom questions.",
        "guided_done": "Thanks, that's everything I need for the classification score.",
    },
    "hi": {
        "app_title": "🩺 रुमेटीइड आर्थराइटिस RAG सहायक",
        "app_subtitle": "सवाल पूछें, एक्स-रे अपलोड करें, और एक आधारित वर्गीकरण सुझाव पाने के लिए कुछ लक्षण संबंधी सवालों के जवाब दें।",
        "language_label": "भाषा",
        "xray_header": "📷 एक्स-रे (वैकल्पिक)",
        "xray_upload_label": "हाथ/कलाई का एक्स-रे अपलोड करें",
        "found_cases": "{n} समान संदर्भ मामले मिले।",
        "no_classifier": "कोई फाइन-ट्यून किया गया क्लासिफायर नहीं मिला -- इसे सक्षम करने के लिए scripts/finetune_vit.py चलाएं।",
        "symptom_header": "📋 लक्षण जांच (ACR/EULAR)",
        "symptom_caption": "एक आधारित वर्गीकरण स्कोर के लिए इनका उत्तर दें, या नीचे दिए गए निर्देशित चैट बटन का उपयोग करें।",
        "symptom_small": "छोटे जोड़ (उंगलियां/कलाई) सूजे/दर्दयुक्त",
        "symptom_large": "बड़े जोड़ (कंधे/कोहनी/घुटने/कूल्हे) सूजे/दर्दयुक्त",
        "symptom_rf": "RF / एंटी-CCP रक्त परीक्षण परिणाम",
        "symptom_crp": "CRP/ESR असामान्य?",
        "symptom_weeks": "लक्षणों की अवधि (सप्ताह)",
        "submit_button": "लक्षण जमा करें",
        "guided_chat_button": "🗣️ इसके बजाय मुझसे चैट में ये सवाल पूछें",
        "chat_placeholder": "अपनी स्थिति, लक्षणों या अपलोड किए गए एक्स-रे के बारे में पूछें...",
        "welcome_message": (
            "नमस्ते! मैं आपका RA सहायक हूं। आप मुझसे रुमेटीइड आर्थराइटिस के बारे में कुछ भी पूछ सकते हैं, "
            "साइडबार से एक्स-रे अपलोड कर सकते हैं, या एक आधारित वर्गीकरण सुझाव के लिए लक्षण जांच भर सकते हैं। "
            "आप क्या जानना चाहेंगे?"
        ),
        "dx_positive_label": "ACR/EULAR सीमा को पूरा करता है",
        "dx_negative_label": "ACR/EULAR सीमा से नीचे",
        "dx_cutoff_note": "RA वर्गीकरण के लिए (सीमा: 6/10)।",
        "disclaimer_text": (
            "यह ACR/EULAR स्कोरिंग नियमों पर आधारित एक वर्गीकरण सुझाव है, चिकित्सा निदान नहीं। "
            "कृपया किसी भी परिणाम की पुष्टि रुमेटोलॉजिस्ट से करें।"
        ),
        "voice_input_label": "🎤 या अपना सवाल रिकॉर्ड करें",
        "voice_output_toggle": "🔊 उत्तर ज़ोर से पढ़ें",
        "transcribing": "आपका वॉइस संदेश ट्रांसक्राइब हो रहा है...",
        "rf_options": {"negative": "नेगेटिव", "low_positive": "कम पॉज़िटिव", "high_positive": "अधिक पॉज़िटिव"},
        "crp_options": {"no": "नहीं", "yes": "हां"},
        "prediction_label": "पूर्वानुमान: **{label}** ({confidence} विश्वास)",
        "gradcam_caption": "Grad-CAM: मॉडल ने कहाँ ध्यान केंद्रित किया",
        "spinner_analyzing": "छवि का विश्लेषण और समान मामलों की खोज हो रही है...",
        "spinner_classifier": "क्लासिफायर चलाया जा रहा है और Grad-CAM बनाया जा रहा है...",
        "spinner_answer": "जानकारी खोजी जा रही है और उत्तर तैयार किया जा रहा है...",
        "guided_intro": "ज़रूर -- चलिए कुछ त्वरित लक्षण संबंधी सवाल पूछते हैं।",
        "guided_done": "धन्यवाद, वर्गीकरण स्कोर के लिए मुझे बस इतना ही चाहिए था।",
    },
    "te": {
        "app_title": "🩺 రుమటాయిడ్ ఆర్థరైటిస్ RAG సహాయకుడు",
        "app_subtitle": "ప్రశ్నలు అడగండి, ఎక్స్-రే అప్‌లోడ్ చేయండి, మరియు ఆధారిత వర్గీకరణ సూచన కోసం కొన్ని లక్షణాల ప్రశ్నలకు సమాధానం ఇవ్వండి.",
        "language_label": "భాష",
        "xray_header": "📷 ఎక్స్-రే (ఐచ్ఛికం)",
        "xray_upload_label": "చేతి/మణికట్టు ఎక్స్-రే అప్‌లోడ్ చేయండి",
        "found_cases": "{n} సారూప్య రిఫరెన్స్ కేసులు కనుగొనబడ్డాయి.",
        "no_classifier": "ఫైన్-ట్యూన్ చేసిన క్లాసిఫైయర్ కనుగొనబడలేదు -- దీన్ని ప్రారంభించడానికి scripts/finetune_vit.py నడపండి.",
        "symptom_header": "📋 లక్షణాల తనిఖీ (ACR/EULAR)",
        "symptom_caption": "ఆధారిత వర్గీకరణ స్కోర్ కోసం వీటికి సమాధానం ఇవ్వండి, లేదా దిగువ గైడెడ్ చాట్ బటన్‌ని ఉపయోగించండి.",
        "symptom_small": "చిన్న కీళ్ళు (వేళ్లు/మణికట్టు) వాచి/నొప్పిగా ఉన్నాయా",
        "symptom_large": "పెద్ద కీళ్ళు (భుజాలు/మోచేతులు/మోకాళ్ళు/తుంటి) వాచి/నొప్పిగా ఉన్నాయా",
        "symptom_rf": "RF / యాంటీ-CCP రక్త పరీక్ష ఫలితం",
        "symptom_crp": "CRP/ESR అసాధారణంగా ఉందా?",
        "symptom_weeks": "లక్షణాల వ్యవధి (వారాలు)",
        "submit_button": "లక్షణాలను సమర్పించండి",
        "guided_chat_button": "🗣️ బదులుగా చాట్‌లో ఈ ప్రశ్నలు అడగండి",
        "chat_placeholder": "మీ పరిస్థితి, లక్షణాలు లేదా అప్‌లోడ్ చేసిన ఎక్స్-రే గురించి అడగండి...",
        "welcome_message": (
            "నమస్తే! నేను మీ RA సహాయకుడిని. మీరు రుమటాయిడ్ ఆర్థరైటిస్ గురించి ఏదైనా అడగవచ్చు, "
            "సైడ్‌బార్ నుండి ఎక్స్-రే అప్‌లోడ్ చేయవచ్చు, లేదా ఆధారిత వర్గీకరణ సూచన కోసం లక్షణాల తనిఖీని పూరించవచ్చు. "
            "మీరు ఏమి తెలుసుకోవాలనుకుంటున్నారు?"
        ),
        "dx_positive_label": "ACR/EULAR పరిమితిని చేరుకుంది",
        "dx_negative_label": "ACR/EULAR పరిమితి కంటే తక్కువ",
        "dx_cutoff_note": "RA వర్గీకరణ కోసం (కటాఫ్: 6/10).",
        "disclaimer_text": (
            "ఇది ACR/EULAR స్కోరింగ్ నియమాల ఆధారంగా ఒక వర్గీకరణ సూచన మాత్రమే, వైద్య నిర్ధారణ కాదు. "
            "దయచేసి ఏదైనా ఫలితాన్ని రుమటాలజిస్ట్‌తో నిర్ధారించుకోండి."
        ),
        "voice_input_label": "🎤 లేదా మీ ప్రశ్నను రికార్డ్ చేయండి",
        "voice_output_toggle": "🔊 సమాధానాలను గట్టిగా చదవండి",
        "transcribing": "మీ వాయిస్ సందేశం లిప్యంతరీకరించబడుతోంది...",
        "rf_options": {"negative": "నెగటివ్", "low_positive": "తక్కువ పాజిటివ్", "high_positive": "అధిక పాజిటివ్"},
        "crp_options": {"no": "కాదు", "yes": "అవును"},
        "prediction_label": "అంచనా: **{label}** ({confidence} నమ్మకం)",
        "gradcam_caption": "Grad-CAM: మోడల్ దృష్టి పెట్టిన ప్రాంతం",
        "spinner_analyzing": "చిత్రాన్ని విశ్లేషిస్తూ సారూప్య కేసులను వెతుకుతోంది...",
        "spinner_classifier": "వర్గీకరణదారుని అమలు చేసి Grad-CAM తయారు చేస్తోంది...",
        "spinner_answer": "సమాచారాన్ని వెతికి సమాధానం తయారు చేస్తోంది...",
        "guided_intro": "తప్పకుండా -- కొన్ని త్వరిత లక్షణాల ప్రశ్నలు అడుగుతాను.",
        "guided_done": "ధన్యవాదాలు, వర్గీకరణ స్కోర్ కోసం నాకు కావాల్సినది అంతే.",
    },
}


def t(lang_code: str, key: str, **kwargs) -> str:
    """Get a UI string for the given language, falling back to English."""
    value = UI_STRINGS.get(lang_code, UI_STRINGS["en"]).get(key, UI_STRINGS["en"].get(key, key))
    if isinstance(value, str) and kwargs:
        return value.format(**kwargs)
    return value