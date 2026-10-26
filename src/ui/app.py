import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DRClassifier

# --- Advanced CSS for Hospital-Grade UI ---
st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Header styling */
    .main-header {
        background-color: #0F172A;
        padding: 2rem;
        border-radius: 0 0 20px 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Card-style containers */
    .clinical-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Result badges */
    .severity-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        color: white;
    }

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F1F5F9 !important;
        border-right: 1px solid #CBD5E1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="RetinaScan AI | Rural Diagnostic Suite",
    page_icon="🩺",
    layout="wide"
)

# --- Professional Header ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.5rem;'>🩺 RetinaScan AI</h1>
        <p style='margin:0; opacity: 0.8; font-size: 1.1rem;'>Advanced Explainable AI for Diabetic Retinopathy Screening in Rural Areas</p>
    </div>
    """, unsafe_allow_html=True)

# --- Load Model and Tools ---
@st.cache_resource
def load_clinical_tools():
    model = DRClassifier()
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        status = "Production Model"
    except:
        status = "Pre-trained Research Model"

    target_layer = model.base_model.features[-1]
    from src.xai.explainer import DRExplainer
    explainer = DRExplainer(model, target_layer)
    processor = DRImageProcessor()
    return model, explainer, processor, status

model, explainer, processor, model_status = load_clinical_tools()

# --- Sidebar: Clinical Administration ---
with st.sidebar:
    st.markdown("### 👤 Patient Registry")
    with st.container():
        patient_id = st.text_input("Patient ID", "RS-2026-8842")
        col_a, col_b = st.columns(2)
        age = col_a.number_input("Age", 0, 120, 62)
        gender = col_b.selectbox("Gender", ["Male", "Female", "Other"])
        clinic = st.text_input("Facility", "Rural Health Outpost A")

    st.markdown("---")
    st.markdown("### ⚙️ System Parameters")
    conf_thresh = st.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("---")
    st.markdown(f"**System Status:** `{model_status}`")
    st.markdown("<p style='font-size:0.8rem; color:gray;'>v2.1.0 Stable | GPU: CPU-Inference</p>", unsafe_allow_html=True)

# --- Main Interface ---
st.markdown("### 📥 Image Acquisition")
input_mode = st.radio("Select Input Method", ["Upload Image", "Capture from Camera"], horizontal=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload High-Resolution Fundus Image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo of the retinal scan")

with col_info:

    st.markdown("""
    <div class="clinical-card">
        <h4 style='margin-top:0;'>📋 Scanning Protocol</h4>
        <ul style='font-size:0.9rem; color:#475569;'>
            <li>Ensure image is centered on the optic disc.</li>
            <li>Avoid glare or obstructive artifacts.</li>
            <li>Minimum resolution: 512x512px recommended.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    # Process Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Analysis Phase
    with st.spinner("🔍 Performing Neural Feature Analysis..."):
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_img = processor.preprocess(temp_path)
        input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        heatmap = explainer.generate_heatmap(input_tensor)
        explanation_img = explainer.apply_heatmap_to_image(heatmap, img_array)

    # Visualization Layout
    st.markdown("### 🔬 Diagnostic Analysis")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("<p style='text-align:center; font-weight:bold;'>Original Fundus Image</p>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with viz_col2:
        st.markdown("<p style='text-align:center; font-weight:bold;'>Pathological Saliency Map</p>", unsafe_allow_html=True)
        st.image(explanation_img, use_container_width=True)

    # Final Clinical Report
    st.markdown("---")
    st.markdown("### 📑 Final Clinical Assessment")

    dr_labels = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
    colors = {0: "#10B981", 1: "#3B82F6", 2: "#F59E0B", 3: "#EF4444", 4: "#7F1D1D"}

    severity = dr_labels[pred.item()]
    color = colors[pred.item()]
    action = "🚨 Urgent Specialist Referral" if pred.item() >= 3 else "✅ Routine follow-up (6-12 months)"

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Severity Level", severity)
    with m2:
        st.metric("AI Confidence", f"{conf.item()*100:.1f}%")
    with m3:
        st.metric("Action", "Referral" if pred.item() >= 3 else "Monitor")

    # Report Box
    st.markdown(f"""
    <div class="clinical-card" style="border-left: 8px solid {color};">
        <h3 style='color:{color}; margin-top:0;'>Diagnostic Summary</h3>
        <p style='font-size:1.1rem;'>
            Patient <b>{patient_id}</b> presents with features consistent with <b>{severity}</b>.
        </p>
        <div style="background-color:#F1F5F9; padding:15px; border-radius:8px; margin:10px 0;">
            <b>Recommendation:</b> {action}
        </div>
        <p style='font-size:0.85rem; color:gray; font-style:italic;'>
            Note: This AI tool is for screening support. Final diagnosis must be confirmed by a certified ophthalmologist.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem; color:#64748B;">
        <h2 style="font-weight:300;">Ready for Analysis</h2>
        <p>Please upload a fundus image to begin the diagnostic process.</p>
    </div>
    """, unsafe_allow_html=True)
