import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DREnsembleClassifier as DRClassifier
import base64

# --- Professional CSS for Enterprise Clinical Suite ---
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 2rem; border-radius: 0 0 25px 25px; color: white;
        text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
    }
    .clinical-card {
        background-color: white; padding: 1.5rem; border-radius: 15px;
        border: 1px solid #E2E8F0; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .report-header {
        border-bottom: 2px solid #0F172A; padding-bottom: 10px; margin-bottom: 20px;
    }
    .status-pill {
        padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;
        color: white; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(page_title="RetinaScan AI | Enterprise Suite", page_icon="🏥", layout="wide")

# --- Multilingual Support ---
LANGUAGES = {
    "English": {
        "title": "RetinaScan AI", "subtitle": "Enterprise DR Diagnostic Suite", "patient_reg": "👤 Patient Registry",
        "id": "Patient ID", "age": "Age", "gender": "Gender", "facility": "Clinic/Facility",
        "settings": "⚙️ System Settings", "thresh": "Sensitivity", "status": "System Status",
        "acquisition": "📥 Image Acquisition", "mode_select": "Select Input Method",
        "upload": "Upload Image", "camera": "📸 Take Photo from Camera", "protocol": "📋 Scanning Guide",
        "p1": "Center the image on the eye.", "p2": "Ensure good lighting.", "p3": "Hold steady and avoid blur.",
        "analysis": "🔬 AI Analysis", "orig_label": "Original Scan", "xai_label": "AI Explanation (Heatmap)",
        "report": "📑 Diagnostic Report", "severity": "DR Severity", "conf": "AI Confidence", "action": "Recommended Action",
        "note": "Clinical Interpretation", "note_detail": "The AI highlighted areas in red to show where the disease is. Please verify with a doctor.",
        "urgent": "🚨 Urgent: Visit Eye Specialist", "routine": "✅ Routine: Normal Monitoring",
        "no_img": "Please upload or take a photo to begin.", "tip": "Tip: Clear images give better results.",
        "download_pdf": "📥 Download Medical Report (PDF)"
    },
    "Hindi (हिन्दी)": {
        "title": "रेटिनास्कैन AI", "subtitle": "मधुमेह रेटिनोपैथी स्क्रीनिंग", "patient_reg": "👤 रोगी पंजीकरण",
        "id": "रोगी आईडी", "age": "आयु", "gender": "लिंग", "facility": "स्वास्थ्य केंद्र",
        "settings": "⚙️ सिस्टम सेटिंग्स", "thresh": "संवेदनशीलता", "status": "सिस्टम स्थिति",
        "acquisition": "📥 फोटो लें/अपलोड करें", "mode_select": "विधि चुनें",
        "upload": "फोटो अपलोड करें", "camera": "📸 कैमरे से फोटो लें", "protocol": "📋 स्कैनिंग गाइड",
        "p1": "फोटो को आंख के बीचों-बीच रखें।", "p2": "रोशनी अच्छी होनी चाहिए।", "p3": "स्थिर रहें और धुंधलापन न होने दें।",
        "analysis": "🔬 AI विश्लेषण", "orig_label": "मूल फोटो", "xai_label": "AI स्पष्टीकरण (हीटमैप)",
        "report": "📑 जांच रिपोर्ट", "severity": "बीमारी का स्तर", "conf": "AI विश्वास", "action": "सुझाव",
        "note": "क्लिनिकल व्याख्या", "note_detail": "AI ने लाल रंग में उन क्षेत्रों को दिखाया है जहां बीमारी हो सकती है। कृपया डॉक्टर से पुष्टि करें।",
        "urgent": "🚨 तत्काल: नेत्र विशेषज्ञ से मिलें", "routine": "✅ नियमित जांच जारी रखें",
        "no_img": "शुरू करने के लिए कृपया फोटो अपलोड करें या लें।", "tip": "टिप: स्पष्ट फोटो से परिणाम बेहतर आते हैं।",
        "download_pdf": "📥 मेडिकल रिपोर्ट डाउनलोड करें (PDF)"
    }
}

selected_lang = st.sidebar.selectbox("🌐 Language / भाषा", list(LANGUAGES.keys()))
L = LANGUAGES[selected_lang]

# --- Header ---
st.markdown(f"""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.8rem;'>🩺 {L['title']}</h1>
        <p style='margin:0; opacity: 0.9; font-size: 1.2rem;'>{L['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Load Model and Tools ---
@st.cache_resource
def load_clinical_tools():
    model = DRClassifier()
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        status = "Production Ensemble"
    except:
        status = "Pre-trained Ensemble"

    target_layer = model.model_eff.net.features[-1] # Use EfficientNet layer for heatmap
    from src.xai.explainer import DRExplainer
    explainer = DRExplainer(model.model_eff, target_layer) # Explain based on the lead model
    processor = DRImageProcessor()
    return model, explainer, processor, status

model, explainer, processor, model_status = load_clinical_tools()

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"### {L['patient_reg']}")
    p_id = st.text_input(L['id'], "RS-2026-8842")
    col_a, col_b = st.columns(2)
    age = col_a.number_input(L['age'], 0, 120, 62)
    gender = col_b.selectbox(L['gender'], ["Male", "Female", "Other"])
    clinic = st.text_input(L['facility'], "Rural Health Center")
    st.markdown("---")
    st.markdown(f"### {L['settings']}")
    conf_thresh = st.slider(L['thresh'], 0.0, 1.0, 0.5)
    st.markdown(f"**{L['status']}:** `{model_status}`")

# --- Main Interface ---
col_input, col_guide = st.columns([2, 1])
with col_input:
    st.markdown(f"### {L['acquisition']}")
    input_mode = st.radio(L['mode_select'], ["Upload Image", "Capture from Camera"], horizontal=True)
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader(L['upload'], type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input(L['camera'])

with col_guide:
    st.markdown(f"""
    <div class="clinical-card">
        <h4 style='margin-top:0;'>{L['protocol']}</h4>
        <ul style='font-size:0.95rem; color:#475569; line-height:1.6;'><li>{L['p1']}</li><li>{L['p2']}</li><li>{L['p3']}</li></ul>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("🔍 Ensemble Analysis in progress..."):
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
        processed_img = processor.preprocess(temp_path)
        input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        heatmap = explainer.generate_heatmap(input_tensor)
        explanation_img = explainer.apply_heatmap_to_image(heatmap, img_array)

    st.markdown(f"### {L['analysis']}")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown(f"<p style='text-align:center;'>{L['orig_label']}</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with v2:
        st.markdown(f"<p style='text-align:center;'>{L['xai_label']}</p>", unsafe_allow_html=True)
        st.image(explanation_img, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### {L['report']}")

    dr_labels = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
    colors = {0: "#10B981", 1: "#3B82F6", 2: "#F59E0B", 3: "#EF4444", 4: "#7F1D1D"}
    severity = dr_labels[pred.item()]
    color = colors[pred.item()]
    action = L['urgent'] if pred.item() >= 3 else L['routine']

    m1, m2, m3 = st.columns(3)
    with m1: st.metric(L['severity'], severity)
    with m2: st.metric(L['conf'], f"{conf.item()*100:.1f}%")
    with m3: st.metric(L['action'], "Urgent" if pred.item() >= 3 else "Normal")

    # Report box
    st.markdown(f"""
    <div class="clinical-card" style="border-left: 10px solid {color};">
        <h3 style='color:{color}; margin-top:0;'>{L['note']}</h3>
        <p style='font-size:1.2rem;'>Patient <b>{p_id}</b>: <b style='color:{color};'>{severity}</b></p>
        <div style="background-color:#F1F5F9; padding:15px; border-radius:10px; margin:15px 0; font-size:1.1rem;">
            <b>{L['action']}:</b> {action}
        </div>
        <p style='font-size:0.9rem; color:gray;'>{L['note_detail']}</p>
    </div>
    """, unsafe_allow_html=True)

    # PDF Generation (Simplified as HTML download for Streamlit)
    pdf_content = f"""
    <html>
    <body style="font-family: sans-serif; padding: 40px;">
        <div style="text-align:center; border-bottom: 2px solid black;">
            <h1>RetinaScan AI Medical Report</h1>
            <p>Clinic: {clinic} | Date: 2026-10-27</p>
        </div>
        <h3>Patient Information</h3>
        <p>ID: {p_id} | Age: {age} | Gender: {gender}</p>
        <h3>Diagnostic Result</h3>
        <p><b>Severity:</b> {severity}</p>
        <p><b>Confidence:</b> {conf.item()*100:.1f}%</p>
        <p><b>Recommended Action:</b> {action}</p>
        <p><i>This report was generated by AI and must be verified by a clinician.</i></p>
    </body>
    </html>
    """
    b64 = base64.b64encode(pdf_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="Report_{p_id}.html"> {L["download_pdf"]} </a>'
    st.markdown(f'<div style="text-align:right;">{href}</div>', unsafe_allow_html=True)

else:
    st.markdown(f"<div style='text-align:center; padding:5rem; color:#64748B;'><h2>{L['no_img']}</h2><p>{L['tip']}</p></div>", unsafe_allow_html=True)
