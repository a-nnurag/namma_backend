"""
Per-skill system prompts for Groq llama-3.3-70b-versatile.

Kept under 500 tokens each. Directive phrasing guides the model to ask
exactly the right questions in order, then end gracefully.

Supported languages: kn (Kannada), hi (Hindi), en (English), te (Telugu), ta (Tamil)
"""

_LANGUAGE_NAMES: dict[str, str] = {
    "kn": "Kannada (ಕನ್ನಡ)",
    "hi": "Hindi (हिन्दी)",
    "en": "English",
    "te": "Telugu (తెలుగు)",
    "ta": "Tamil (தமிழ்)",
}

_BASE_TEMPLATE = """\
You are a friendly interviewer for NammaKelsa, a job placement platform for skilled workers in India.
You are conducting a spoken interview for a candidate applying as a {skill_name}.

LANGUAGE RULE — CRITICAL:
- The candidate speaks {language_name}. You MUST respond ONLY in {language_name}.
- Do NOT switch to English unless the candidate writes in English.
- Use simple, conversational vocabulary appropriate for a skilled trade worker.

Interview rules:
- Ask ONE question at a time. Keep it SHORT (1 sentence max).
- After {num_questions} questions, say exactly in {language_name}: "{closing_message}"
- Do not evaluate or judge. Just ask questions and listen.
- Silently fix obvious STT errors in names or places.
- Maximum 2 sentences per response.
- Start immediately with the first question. Do not say hello or introduce yourself.

Questions to ask in order (adapt wording based on candidate's answers):
{questions}
"""

_CLOSING: dict[str, str] = {
    "kn": "ಧನ್ಯವಾದಗಳು! ನಾವು ಶೀಘ್ರದಲ್ಲೇ ನಿಮ್ಮನ್ನು ಸಂಪರ್ಕಿಸುತ್ತೇವೆ.",
    "hi": "धन्यवाद! हम जल्द ही आपसे संपर्क करेंगे।",
    "en": "Thank you, that's all for now! We will get back to you soon.",
    "te": "ధన్యవాదాలు! మేము త్వరలో మీకు తెలియజేస్తాము.",
    "ta": "நன்றி! நாங்கள் விரைவில் உங்களை தொடர்பு கொள்வோம்.",
}

# ── Skill questions: English + Kannada translations ───────────────────────────

_SKILL_QUESTIONS: dict[str, dict[str, list[str]]] = {
    "electrician": {
        "en": [
            "How many years have you worked as an electrician?",
            "What types of wiring work have you done — domestic, industrial, or both?",
            "How do you stay safe when working with live wires?",
            "What tools do you use every day on the job?",
            "Have you worked with 3-phase connections?",
            "What do you do if you find a short circuit?",
            "Can you read electrical diagrams or drawings?",
            "Tell me about a big project you worked on.",
        ],
        "kn": [
            "ನೀವು ಎಷ್ಟು ವರ್ಷ ಎಲೆಕ್ಟ್ರಿಷಿಯನ್ ಆಗಿ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ?",
            "ನೀವು ಯಾವ ರೀತಿಯ ವೈರಿಂಗ್ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ — ಮನೆ, ಕಾರ್ಖಾನೆ, ಅಥವಾ ಎರಡೂ?",
            "ಜೀವಂತ ತಂತಿಗಳೊಂದಿಗೆ ಕೆಲಸ ಮಾಡುವಾಗ ನೀವು ಸುರಕ್ಷಿತವಾಗಿ ಇರಲು ಏನು ಮಾಡುತ್ತೀರಿ?",
            "ನೀವು ಪ್ರತಿದಿನ ಯಾವ ಉಪಕರಣಗಳನ್ನು ಬಳಸುತ್ತೀರಿ?",
            "ನೀವು 3-ಫೇಸ್ ಸಂಪರ್ಕಗಳೊಂದಿಗೆ ಕೆಲಸ ಮಾಡಿದ್ದೀರಾ?",
            "ಶಾರ್ಟ್ ಸರ್ಕ್ಯೂಟ್ ಕಂಡುಬಂದರೆ ನೀವು ಏನು ಮಾಡುತ್ತೀರಿ?",
            "ನೀವು ವಿದ್ಯುತ್ ನಕ್ಷೆಗಳನ್ನು ಓದಬಲ್ಲಿರಾ?",
            "ನೀವು ಕೆಲಸ ಮಾಡಿದ ದೊಡ್ಡ ಯೋಜನೆಯ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आप कितने साल से इलेक्ट्रीशियन का काम कर रहे हैं?",
            "आपने किस तरह की वायरिंग की है — घरेलू, औद्योगिक या दोनों?",
            "लाइव तारों के साथ काम करते समय आप सुरक्षित कैसे रहते हैं?",
            "आप रोज़ाना कौन से उपकरण इस्तेमाल करते हैं?",
            "क्या आपने 3-फेज़ कनेक्शन पर काम किया है?",
            "शॉर्ट सर्किट मिलने पर आप क्या करते हैं?",
            "क्या आप इलेक्ट्रिकल डायग्राम पढ़ सकते हैं?",
            "अपने किसी बड़े प्रोजेक्ट के बारे में बताइए।",
        ],
    },
    "plumber": {
        "en": [
            "How many years have you worked as a plumber?",
            "What kind of plumbing work have you done — residential or commercial?",
            "What pipes do you usually work with — PVC, CPVC, or iron?",
            "How do you find where a water leak is?",
            "Have you installed bathroom fixtures before?",
            "How do you handle a blocked drain?",
            "Do you know how to read plumbing drawings?",
            "Tell me about a difficult job you solved.",
        ],
        "kn": [
            "ನೀವು ಎಷ್ಟು ವರ್ಷ ಪ್ಲಂಬರ್ ಆಗಿ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ?",
            "ನೀವು ಯಾವ ರೀತಿಯ ಪ್ಲಂಬಿಂಗ್ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ — ಮನೆ ಅಥವಾ ವಾಣಿಜ್ಯ?",
            "ನೀವು ಸಾಮಾನ್ಯವಾಗಿ ಯಾವ ಪೈಪ್ ಬಳಸುತ್ತೀರಿ — PVC, CPVC, ಅಥವಾ ಕಬ್ಬಿಣ?",
            "ನೀರು ಸೋರಿಕೆ ಎಲ್ಲಿದೆ ಎಂದು ನೀವು ಹೇಗೆ ಕಂಡುಹಿಡಿಯುತ್ತೀರಿ?",
            "ನೀವು ಬಾತ್‌ರೂಮ್ ಫಿಟ್ಟಿಂಗ್ಸ್ ಅಳವಡಿಸಿದ್ದೀರಾ?",
            "ಚರಂಡಿ ಮುಚ್ಚಿದ್ದರೆ ನೀವು ಏನು ಮಾಡುತ್ತೀರಿ?",
            "ನೀವು ಪ್ಲಂಬಿಂಗ್ ನಕ್ಷೆಗಳನ್ನು ಓದಬಲ್ಲಿರಾ?",
            "ನೀವು ಪರಿಹರಿಸಿದ ಕಷ್ಟಕರ ಕೆಲಸದ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आप कितने साल से प्लंबर का काम कर रहे हैं?",
            "आपने किस तरह का प्लंबिंग काम किया है — घरेलू या व्यावसायिक?",
            "आप आमतौर पर कौन से पाइप इस्तेमाल करते हैं — PVC, CPVC या लोहे के?",
            "पानी का रिसाव कहाँ है यह आप कैसे पता लगाते हैं?",
            "क्या आपने बाथरूम फिटिंग्स लगाई हैं?",
            "बंद नाले को आप कैसे ठीक करते हैं?",
            "क्या आप प्लंबिंग के नक्शे पढ़ सकते हैं?",
            "किसी मुश्किल काम के बारे में बताइए जो आपने सुलझाया।",
        ],
    },
    "welder": {
        "en": [
            "How many years have you been welding?",
            "What welding types can you do — MIG, TIG, Arc, or Gas?",
            "What metals have you welded?",
            "What safety equipment do you always wear while welding?",
            "How do you check the quality of a weld?",
            "Have you worked on structural welding or only small jobs?",
            "Can you read welding blueprints?",
            "Tell me about your most challenging welding job.",
        ],
        "kn": [
            "ನೀವು ಎಷ್ಟು ವರ್ಷಗಳಿಂದ ವೆಲ್ಡಿಂಗ್ ಮಾಡುತ್ತಿದ್ದೀರಿ?",
            "ನೀವು ಯಾವ ರೀತಿಯ ವೆಲ್ಡಿಂಗ್ ಮಾಡಬಲ್ಲಿರಿ — MIG, TIG, Arc, ಅಥವಾ Gas?",
            "ನೀವು ಯಾವ ಲೋಹಗಳನ್ನು ವೆಲ್ಡ್ ಮಾಡಿದ್ದೀರಿ?",
            "ವೆಲ್ಡಿಂಗ್ ಮಾಡುವಾಗ ನೀವು ಯಾವ ಸುರಕ್ಷಾ ಉಪಕರಣಗಳನ್ನು ತೊಡುತ್ತೀರಿ?",
            "ವೆಲ್ಡ್ ಗುಣಮಟ್ಟವನ್ನು ನೀವು ಹೇಗೆ ಪರೀಕ್ಷಿಸುತ್ತೀರಿ?",
            "ನೀವು ರಚನಾತ್ಮಕ ವೆಲ್ಡಿಂಗ್ ಮಾಡಿದ್ದೀರಾ ಅಥವಾ ಸಣ್ಣ ಕೆಲಸಗಳನ್ನು ಮಾತ್ರ?",
            "ನೀವು ವೆಲ್ಡಿಂಗ್ ನೀಲನಕ್ಷೆಗಳನ್ನು ಓದಬಲ್ಲಿರಾ?",
            "ನಿಮ್ಮ ಅತ್ಯಂತ ಕಷ್ಟಕರವಾದ ವೆಲ್ಡಿಂಗ್ ಕೆಲಸದ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आप कितने सालों से वेल्डिंग कर रहे हैं?",
            "आप कौन सी वेल्डिंग कर सकते हैं — MIG, TIG, Arc या Gas?",
            "आपने किन धातुओं पर वेल्डिंग की है?",
            "वेल्डिंग करते समय आप कौन सा सुरक्षा उपकरण पहनते हैं?",
            "वेल्ड की गुणवत्ता आप कैसे जांचते हैं?",
            "क्या आपने स्ट्रक्चरल वेल्डिंग की है या सिर्फ छोटे काम?",
            "क्या आप वेल्डिंग के ब्लूप्रिंट पढ़ सकते हैं?",
            "अपने सबसे मुश्किल वेल्डिंग काम के बारे में बताइए।",
        ],
    },
    "carpenter": {
        "en": [
            "How many years have you worked as a carpenter?",
            "What kind of work do you mostly do — furniture, doors, roofing, or formwork?",
            "What wood types do you work with?",
            "What are your main tools?",
            "How do you measure and cut wood accurately?",
            "Have you made furniture or done site carpentry?",
            "Can you read drawings or sketches?",
            "Tell me about a project you are proud of.",
        ],
        "kn": [
            "ನೀವು ಎಷ್ಟು ವರ್ಷ ಬಡಗಿ ಆಗಿ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ?",
            "ನೀವು ಹೆಚ್ಚಾಗಿ ಯಾವ ಕೆಲಸ ಮಾಡುತ್ತೀರಿ — ಪೀಠೋಪಕರಣ, ಬಾಗಿಲು, ಮೇಲ್ಛಾವಣಿ, ಅಥವಾ ಫಾರ್ಮ್‌ವರ್ಕ್?",
            "ನೀವು ಯಾವ ರೀತಿಯ ಮರದೊಂದಿಗೆ ಕೆಲಸ ಮಾಡುತ್ತೀರಿ?",
            "ನಿಮ್ಮ ಮುಖ್ಯ ಉಪಕರಣಗಳು ಯಾವುವು?",
            "ನೀವು ಮರವನ್ನು ಸರಿಯಾಗಿ ಅಳತೆ ಮಾಡಿ ಕತ್ತರಿಸುವುದು ಹೇಗೆ?",
            "ನೀವು ಪೀಠೋಪಕರಣ ತಯಾರಿಸಿದ್ದೀರಾ ಅಥವಾ ಸೈಟ್ ಕಾರ್ಪೆಂಟ್ರಿ ಮಾಡಿದ್ದೀರಾ?",
            "ನೀವು ನಕ್ಷೆ ಅಥವಾ ಸ್ಕೆಚ್ ಓದಬಲ್ಲಿರಾ?",
            "ನೀವು ಹೆಮ್ಮೆಪಡುವ ಯೋಜನೆಯ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आप कितने साल से बढ़ई का काम कर रहे हैं?",
            "आप ज्यादातर कौन सा काम करते हैं — फर्नीचर, दरवाजे, छत या फॉर्मवर्क?",
            "आप किस तरह की लकड़ी से काम करते हैं?",
            "आपके मुख्य औजार कौन से हैं?",
            "आप लकड़ी को सही तरीके से कैसे नापते और काटते हैं?",
            "क्या आपने फर्नीचर बनाया है या सिर्फ साइट कारपेंट्री की है?",
            "क्या आप नक्शे या स्केच पढ़ सकते हैं?",
            "किस प्रोजेक्ट पर आपको गर्व है, बताइए।",
        ],
    },
    "mason": {
        "en": [
            "How many years have you been doing masonry work?",
            "What kind of masonry do you do — brickwork, plastering, tiling, or all?",
            "How do you mix mortar correctly?",
            "How do you ensure a wall is straight and level?",
            "What is your experience with different types of bricks or blocks?",
            "Have you worked on large construction sites?",
            "How do you handle cracks in plaster?",
            "Tell me about a construction project you worked on.",
        ],
        "kn": [
            "ನೀವು ಎಷ್ಟು ವರ್ಷಗಳಿಂದ ಗಾರೆ ಕೆಲಸ ಮಾಡುತ್ತಿದ್ದೀರಿ?",
            "ನೀವು ಯಾವ ರೀತಿಯ ಕೆಲಸ ಮಾಡುತ್ತೀರಿ — ಇಟ್ಟಿಗೆ, ಪ್ಲಾಸ್ಟರ್, ಟೈಲ್ಸ್, ಅಥವಾ ಎಲ್ಲವೂ?",
            "ಮೊರ್ಟಾರ್ ಅನ್ನು ಸರಿಯಾಗಿ ಮಿಶ್ರಣ ಮಾಡುವುದು ಹೇಗೆ?",
            "ಗೋಡೆ ನೇರವಾಗಿ ಮತ್ತು ಸಮತಲವಾಗಿರುವಂತೆ ನೀವು ಹೇಗೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳುತ್ತೀರಿ?",
            "ವಿವಿಧ ರೀತಿಯ ಇಟ್ಟಿಗೆ ಅಥವಾ ಬ್ಲಾಕ್‌ಗಳೊಂದಿಗೆ ನಿಮ್ಮ ಅನುಭವ ಏನು?",
            "ನೀವು ದೊಡ್ಡ ನಿರ್ಮಾಣ ತಾಣಗಳಲ್ಲಿ ಕೆಲಸ ಮಾಡಿದ್ದೀರಾ?",
            "ಪ್ಲಾಸ್ಟರ್‌ನಲ್ಲಿ ಬಿರುಕು ಕಂಡುಬಂದರೆ ನೀವು ಏನು ಮಾಡುತ್ತೀರಿ?",
            "ನೀವು ಕೆಲಸ ಮಾಡಿದ ನಿರ್ಮಾಣ ಯೋಜನೆಯ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आप कितने सालों से राजमिस्त्री का काम कर रहे हैं?",
            "आप किस तरह का काम करते हैं — ईंट, प्लास्टर, टाइल्स या सभी?",
            "मोर्टार को सही तरीके से कैसे मिलाते हैं?",
            "दीवार को सीधी और समतल रखने के लिए आप क्या करते हैं?",
            "अलग-अलग तरह की ईंटों या ब्लॉक्स के साथ आपका क्या अनुभव है?",
            "क्या आपने बड़े निर्माण स्थलों पर काम किया है?",
            "प्लास्टर में दरार आने पर आप क्या करते हैं?",
            "किसी निर्माण प्रोजेक्ट के बारे में बताइए जिस पर आपने काम किया।",
        ],
    },
    "labour": {
        "en": [
            "What kind of construction or site work have you done before?",
            "How many years of site work experience do you have?",
            "What heavy tools or machines have you operated?",
            "How do you stay safe on a construction site?",
            "Can you read simple site instructions or signs?",
            "What is the hardest physical work you have done?",
            "Are you comfortable working at heights or in confined spaces?",
            "Tell me about a team you worked in on a site.",
        ],
        "kn": [
            "ನೀವು ಯಾವ ರೀತಿಯ ನಿರ್ಮಾಣ ಅಥವಾ ಸೈಟ್ ಕೆಲಸ ಮಾಡಿದ್ದೀರಿ?",
            "ನಿಮಗೆ ಎಷ್ಟು ವರ್ಷ ಸೈಟ್ ಕೆಲಸದ ಅನುಭವ ಇದೆ?",
            "ನೀವು ಯಾವ ಭಾರೀ ಉಪಕರಣ ಅಥವಾ ಯಂತ್ರ ಬಳಸಿದ್ದೀರಿ?",
            "ನಿರ್ಮಾಣ ತಾಣದಲ್ಲಿ ನೀವು ಸುರಕ್ಷಿತವಾಗಿ ಇರಲು ಏನು ಮಾಡುತ್ತೀರಿ?",
            "ನೀವು ಸರಳ ಸೈಟ್ ಸೂಚನೆಗಳು ಅಥವಾ ಚಿಹ್ನೆಗಳನ್ನು ಓದಬಲ್ಲಿರಾ?",
            "ನೀವು ಮಾಡಿದ ಅತ್ಯಂತ ಕಷ್ಟಕರ ದೈಹಿಕ ಕೆಲಸ ಯಾವುದು?",
            "ನೀವು ಎತ್ತರದಲ್ಲಿ ಅಥವಾ ಸಂಕುಚಿತ ಸ್ಥಳಗಳಲ್ಲಿ ಕೆಲಸ ಮಾಡಲು ಆರಾಮದಾಯಕರಾಗಿದ್ದೀರಾ?",
            "ನೀವು ಸೈಟ್‌ನಲ್ಲಿ ಕೆಲಸ ಮಾಡಿದ ತಂಡದ ಬಗ್ಗೆ ಹೇಳಿ.",
        ],
        "hi": [
            "आपने पहले किस तरह का निर्माण या साइट काम किया है?",
            "आपको साइट काम का कितना अनुभव है?",
            "आपने कौन से भारी उपकरण या मशीनें चलाई हैं?",
            "निर्माण स्थल पर आप सुरक्षित कैसे रहते हैं?",
            "क्या आप सरल साइट निर्देश या संकेत पढ़ सकते हैं?",
            "आपने अब तक का सबसे मुश्किल शारीरिक काम कौन सा किया?",
            "क्या आप ऊंचाई पर या बंद जगहों पर काम करने में सहज हैं?",
            "किसी साइट पर आपने जिस टीम के साथ काम किया उसके बारे में बताइए।",
        ],
    },
}

_GENERIC_QUESTIONS: dict[str, list[str]] = {
    "en": [
        "How many years of experience do you have in this trade?",
        "What kind of work do you do most often?",
        "What tools do you use every day?",
        "How do you ensure safety on the job?",
        "Tell me about a challenging job you completed.",
    ],
    "kn": [
        "ಈ ಕ್ಷೇತ್ರದಲ್ಲಿ ನಿಮಗೆ ಎಷ್ಟು ವರ್ಷ ಅನುಭವ ಇದೆ?",
        "ನೀವು ಹೆಚ್ಚಾಗಿ ಯಾವ ರೀತಿಯ ಕೆಲಸ ಮಾಡುತ್ತೀರಿ?",
        "ನೀವು ಪ್ರತಿದಿನ ಯಾವ ಉಪಕರಣಗಳನ್ನು ಬಳಸುತ್ತೀರಿ?",
        "ಕೆಲಸದಲ್ಲಿ ಸುರಕ್ಷತೆ ಖಚಿತಪಡಿಸಲು ನೀವು ಏನು ಮಾಡುತ್ತೀರಿ?",
        "ನೀವು ಪೂರ್ಣಗೊಳಿಸಿದ ಒಂದು ಕಷ್ಟಕರ ಕೆಲಸದ ಬಗ್ಗೆ ಹೇಳಿ.",
    ],
    "hi": [
        "इस काम में आपको कितने साल का अनुभव है?",
        "आप सबसे अधिक किस तरह का काम करते हैं?",
        "आप रोज़ाना कौन से औजार इस्तेमाल करते हैं?",
        "काम पर सुरक्षा कैसे सुनिश्चित करते हैं?",
        "किसी मुश्किल काम के बारे में बताइए जो आपने पूरा किया।",
    ],
}


def get_system_prompt(skill_name: str, language: str = "kn") -> str:
    """
    Build the Groq system prompt for the given skill and candidate language.

    Args:
        skill_name : e.g. "electrician", "plumber"
        language   : BCP-47 short code — "kn", "hi", "en", "te", "ta"
    """
    lang = language if language in _LANGUAGE_NAMES else "kn"
    skill_key = skill_name.lower()

    skill_bank = _SKILL_QUESTIONS.get(skill_key, _GENERIC_QUESTIONS)

    # Pick questions in the requested language; fall back to English
    questions = skill_bank.get(lang) or skill_bank.get("en") or _GENERIC_QUESTIONS["en"]

    numbered        = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    closing_message = _CLOSING.get(lang, _CLOSING["en"])
    language_name   = _LANGUAGE_NAMES.get(lang, "Kannada (ಕನ್ನಡ)")

    return _BASE_TEMPLATE.format(
        skill_name=skill_name,
        language_name=language_name,
        num_questions=len(questions),
        closing_message=closing_message,
        questions=numbered,
    )
