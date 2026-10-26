import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DREnsembleClassifier as DRClassifier
import base64

# --- Ultra-Professional Enterprise CSS ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #F0F4F8 0%, #D9E2EC 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Glassmorphism Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        margin-bottom: 20px;
    }

    /* Top Professional Header */
    .top-header {
        background: #0F172A;
        color: white;
        padding: 1.5rem 3rem;
        border-radius: 0 0 30px 30px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Clinical Severity Badges */
    .badge {
        padding: 6px 15px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        color: white;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Metrics Styling */
    [data-testid="stMetric"] {
        background: white;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #0F172A;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #334155;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Config ---
st.set_page_config(page_title="RetinaScan AI | Clinical Suite", page_icon="🩺", layout="wide")

# --- Language Mapping ---
LANGUAGES = {
    "English": {
        "title": "RetinaScan AI", "subtitle": "Enterprise Diagnostic Suite", "patient_reg": "Patient Registry",
        "id": "Patient ID", "age": "Age", "gender": "Gender", "facility": "Clinic",
        "settings": "System Settings", "thresh": "Sensitivity", "status": "Status",
        "acquisition": "Image Acquisition", "mode_select": "Input Method",
        "upload": "Upload Fundus Image", "camera": "📸 Take Live Photo", "protocol": "Scanning Protocol",
        "p1": "Center image on optic disc.", "p2": "Ensure high brightness.", "p3": "Avoid motion blur.",
        "analysis": "AI Diagnostic Analysis", "orig_label": "Source Image", "xai_label": "AI Feature Saliency",
        "report": "Clinical Findings", "severity": "Severity", "conf": "Confidence", "action": "Recommendation",
        "note": "Doctor's Interpretation", "note_detail": "Heatmap red zones indicate pathology. Verify clinically.",
        "urgent": "Urgent Specialist Referral", "routine": "Standard Monitoring",
        "no_img": "No image uploaded.", "tip": "Upload a high-res scan to begin.",
        "download": "📥 Export Medical Report (PDF)"
    },
    "Hindi (हिन्दी)": {
        "title": "रेटिनास्कैन AI", "subtitle": "एंटरप्राइज डायग्नोस्टिक सुइट", "patient_reg": "रोगी पंजीकरण",
        "id": "रोगी आईडी", "age": "आयु", "gender": "लिंग", "facility": "स्वास्थ्य केंद्र",
        "settings": "सिस्टम सेटिंग्स", "thresh": "संवेदनशीलता", "status": "स्थिति",
        "acquisition": "फोटो अधिग्रहण", "mode_select": "इनपुट विधि",
        "upload": "फोटो अपलोड करें", "camera": "📸 कैमरे से फोटो लें", "protocol": "स्कैनिंग प्रोटोकॉल",
        "p1": "फोटो को आंख के केंद्र में रखें।", "p2": "रोशनी अच्छी होनी चाहिए।", "p3": "धुंधलापन न होने दें।",
        "analysis": "AI विश्लेषण", "orig_label": "मूल फोटो", "xai_label": "AI स्पष्टीकरण (हीटमैप)",
        "report": "जांच रिपोर्ट", "severity": "बीमारी का स्तर", "conf": "AI विश्वास", "action": "सुझाव",
        "note": "क्लिनिकल व्याख्या", "note_detail": "लाल क्षेत्र बीमारी का संकेत देते हैं। डॉक्टर से पुष्टि करें।",
        "urgent": "तत्काल विशेषज्ञ रेफरल", "routine": "नियमित निगरानी",
        "no_img": "शुरू करने के लिए फोटो लें।", "tip": "बेहतर परिणाम के लिए स्पष्ट फोटो लें।",
        "download": "📥 मेडिकल रिपोर्ट डाउनलोड करें"
    }
}

selected_lang = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()))
L = LANGUAGES[selected_lang]

# --- Professional Header ---
st.markdown(f"""
    <div class="top-header">
        <div style="text-align:left;">
            <h1 style='margin:0; font-size: 2rem;'>🩺 {L['title']}</h1>
            <p style='margin:0; opacity: 0.8; font-size: 1rem;'>{L['subtitle']}</p>
        </div>
        <div style="text-align:right;">
            <span class="badge" style="background:#10B981;">SYSTEM ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- Logic and Tools ---
@st.cache_resource
def load_clinical_tools():
    model = DRClassifier()
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        status = "Production"
    except:
        status = "Pre-trained"

    target_layer = model.model_eff.net.features[-1]
    from src.xai.explainer import DRExplainer
    explainer = DRExplainer(model.model_eff, target_layer)
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

# --- Main Application ---
col_main, col_side = st.columns([3, 1])

with col_main:
    st.markdown(f"### {L['acquisition']}")
    input_mode = st.radio(L['mode_select'], ["Upload Image", "Capture from Camera"], horizontal=True)

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader(L['upload'], type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input(L['camera'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("🔬 Analyzing Neural Features..."):
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

    # Diagnostics Section
    st.markdown(f"### {L['analysis']}")
    viz1, viz2 = st.columns(2)
    with viz1:
        st.markdown(f"<p style='text-align:center;'>{L['orig_label']}</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with viz2:
        st.markdown(f"<p style='text-align:center;'>{L['xai_label']}</p>", unsafe_allow_html=True)
        st.image(explanation_img, use_container_width=True)

    # Final Report
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

    st.markdown(f"""
    <div class="glass-card" style="border-left: 12px solid {color};">
        <h3 style='color:{color}; margin-top:0;'>{L['note']}</h3>
        <p style='font-size:1.2rem;'>Patient <b>{p_id}</b>: <b style='color:{color};'>{severity}</b></p>
        <div style="background-color:#F1F5F9; padding:15px; border-radius:12px; margin:15px 0; font-size:1.1rem; color:#1E293B;">
            <b>{L['action']}:</b> {action}
        </div>
        <p style='font-size:0.9rem; color:gray;'>{L['note_detail']}</p>
    </div>
    """, unsafe_allow_html=True)

    # PDF Export
    pdf_content = f"<html><body><h1>RetinaScan Report</h1><p>ID: {p_id}</p><p>Severity: {severity}</p><p>Action: {action}</p></body></html>"
    b64 = base64.b64encode(pdf_content.encode()).decode()
    st.markdown(f'<div style="text-align:right;"><a href="data:text/html;base64,{b64}" download="Report_{p_id}.html" style="text-decoration:none; background:#0F172A; color:white; padding:10px 20px; border-radius:10px;">{L["download"]}</a></div>', unsafe_allow_html=True)

else:
    with col_main:
        st.markdown(f"""
        <div style="text-align:center; padding: 10rem 2rem; color:#64748B;">
            <h2 style="font-weight:300;">{L['no_img']}</h2>
            <p>{L['tip']}</p>
        </div>
        """, unsafe_allow_html=True)

with col_side:
    st.markdown(f"""
    <div class="glass-card">
        <h4 style='margin-top:0;'>{L['protocol']}</h4>
        <ul style='font-size:0.95rem; color:#475569; line-height:1.8;'>
            <li>{L['p1']}</li>
            <li>{L['p2']}</li>
            <li>{L['p3']}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
