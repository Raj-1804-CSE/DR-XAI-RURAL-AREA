import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DREnsembleClassifier as DRClassifier
import base64

# --- LENSKART-INSPIRED PREMIUM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }

    /* Top Navigation Bar */
    .nav-bar {
        background-color: #FFFFFF;
        padding: 1rem 2rem;
        border-bottom: 1px solid #E5E7EB;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
    }

    /* Main Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(to bottom, #F9FAFB, #FFFFFF);
        border-radius: 0 0 50px 50px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }

    /* Lenskart-style Comparison Cards */
    .compare-container {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-top: 20px;
    }

    .compare-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 20px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        width: 45%;
        box-shadow: 0 4px 6px -1px rgba(0, 0,, 0.1);
    }

    .compare-card:hover {
        border-color: #007AFF;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Custom Buttons */
    .stButton>button {
        background-color: #007AFF !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        border: none !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover {
        background-color: #005BB5 !important;
        transform: scale(1.05) !important;
    }

    /* Results Panel */
    .result-panel {
        background-color: #F9FAFB;
        border-radius: 30px;
        padding: 30px;
        margin-top: 40px;
        border: 1px solid #E5E7EB;
    }

    .severity-pill {
        padding: 8px 20px;
        border-radius: 50px;
        font-weight: 600;
        color: white;
        display: inline-block;
        margin: 10px 0;
    }

    /* Sidebar clean up */
    section[data-testid="stSidebar"] {
        background-color: #FBFBFB !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Config ---
st.set_page_config(page_title="RetinaScan AI | Compare & Diagnose", page_icon="👁️", layout="wide")

# --- Languages ---
LANGUAGES = {
    "English": {
        "title": "RetinaScan AI", "subtitle": "Compare your retinal health using AI",
        "patient_reg": "Patient Details", "id": "Patient ID", "age": "Age", "gender": "Gender", "facility": "Clinic",
        "acquisition": "Start Screening", "mode_select": "Choose Method", "upload": "Upload Scan", "camera": "📸 Use Camera",
        "analysis": "Visual Comparison", "orig_label": "Original Scan", "xai_label": "AI Insight",
        "report": "Clinical Report", "severity": "Status", "conf": "Accuracy", "action": "Next Step",
        "download": "Download Report (PDF)"
    },
    "Hindi (हिन्दी)": {
        "title": "रेटिनास्कैन AI", "subtitle": "AI के साथ अपनी आंखों के स्वास्थ्य की जांच करें",
        "patient_reg": "रोगी विवरण", "id": "रोगी आईडी", "age": "आयु", "gender": "लिंग", "facility": "स्वास्थ्य केंद्र",
        "acquisition": "स्क्रीनिंग शुरू करें", "mode_select": "तरीका चुनें", "upload": "फोटो अपलोड करें", "camera": "📸 कैमरा उपयोग करें",
        "analysis": "दृश्य तुलना", "orig_label": "मूल स्कैन", "xai_label": "AI अंतर्दृष्टि",
        "report": "क्लिनिकल रिपोर्ट", "severity": "स्थिति", "conf": "सटीकता", "action": "अगला कदम",
        "download": "रिपोर्ट डाउनलोड करें"
    }
}

selected_lang = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()))
L = LANGUAGES[selected_lang]

# --- Lenskart-style Hero ---
st.markdown(f"""
    <div class="hero-section">
        <h1 class="hero-title">{L['title']}</h1>
        <p class="hero-subtitle">{L['subtitle']}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Load Tools ---
@st.cache_resource
def load_clinical_tools():
    model = DRClassifier()
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    except:
        pass
    target_layer = model.model_eff.net.features[-1]
    from src.xai.explainer import DRExplainer
    explainer = DRExplainer(model.model_eff, target_layer)
    processor = DRImageProcessor()
    return model, explainer, processor

model, explainer, processor = load_clinical_tools()

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"### {L['patient_reg']}")
    p_id = st.text_input(L['id'], "RS-101")
    col_a, col_b = st.columns(2)
    age = col_a.number_input(L['age'], 0, 120, 45)
    gender = col_b.selectbox(L['gender'], ["Male", "Female", "Other"])
    clinic = st.text_input(L['facility'], "Rural Clinic A")

# --- Main Workflow ---
st.markdown(f"<h3 style='text-align:center; margin-top:2rem;'>{L['acquisition']}</h3>", unsafe_allow_html=True)
input_mode = st.radio(L['mode_select'], ["Upload", "Camera"], horizontal=True, label_visibility="collapsed")

col_upload, col_info = st.columns([2, 1])
with col_upload:
    if input_mode == "Upload":
        uploaded_file = st.file_uploader(L['upload'], type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input(L['camera'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    with st.spinner("⚙️ AI analyzing retinal layers..."):
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f: f.write(uploaded_//uploaded_file.getbuffer())
        processed_img = processor.preprocess(temp_path)
        input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        heatmap = explainer.generate_heatmap(input_tensor)
        explanation_img = explainer.apply_heatmap_to_image(heatmap, img_array)

    # --- LENSKART STYLE COMPARISON VIEW ---
    st.markdown(f"<h3 style='text-align:center; margin-top:3rem;'>{L['analysis']}</h3>", unsafe_allow_html=True)

    # Create the comparison row using HTML-like structure in Streamlit
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="compare-card"><p style="font-weight:600; color:#64748B;">{L["orig_label"]}</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="compare-card"><p style="font-weight:600; color:#64748B;">{L["xai_label"]}</p>', unsafe_allow_html=True)
        st.image(explanation_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Result Panel ---
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center;'>{L['report']}</h2>", unsafe_allow_html=True)

    dr_labels = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
    colors = {0: "#10B981", 1: "#3B82F6", 2: "#F59E0B", 3: "#EF4444", 4: "#7F1D1D"}

    severity = dr_labels[pred.item()]
    color = colors[pred.item()]
    action = "Urgent Specialist Referral" if pred.item() >= 3 else "Routine Monitoring"

    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown(f"<div style='text-align:center;'><p>{L['severity']}</p><div class='severity-pill' style='background:{color};'>{severity}</div></div>", unsafe_allow_html=True)
    with res_col2:
        st.markdown(f"<div style='text-align:center;'><p>{L['conf']}</p><div style='font-size:1.5rem; font-weight:bold;'>{conf.item()*100:.1f}%</div></div>", unsafe_allow_html=True)
    with res_col3:
        st.markdown(f"<div style='text-align:center;'><p>{L['action']}</p><div style='font-weight:600; color:#0F172A;'>{action}</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # PDF Download
    pdf_content = f"Patient: {p_id} | Result: {severity} | Action: {action}"
    b64 = base64.b64encode(pdf_content.encode()).decode()
    st.markdown(f'<div style="text-align:center;"><a href="data:text/html;base64,{b64}" download="Report.html" style="text-decoration:none; background:#007AFF; color:white; padding:12px 30px; border-radius:25px; font-weight:600;">{L["download"]}</a></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center; padding: 5rem 2rem; color:#64748B;">
        <h2 style="font-weight:300;">{L['no_img']}</h2>
        <p>{L['tip']}</p>
    </div>
    """, unsafe_allow_html=True)
