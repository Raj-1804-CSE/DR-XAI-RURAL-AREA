import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DRClassifier

# --- Custom CSS for Medical Look ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e4e8;
    }
    .report-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- Page Configuration ---
st.set_page_config(
    page_title="Rural-DR AI | Diagnostic Portal",
    page_icon="🏥",
    layout="wide"
)

# --- Header ---
col_title, col_logo = st.columns([4, 1])
with col_title:
    st.title("🏥 Rural-DR AI Diagnostic Portal")
    st.subheader("Explainable AI for Diabetic Retinopathy Screening")
with col_logo:
    st.markdown("### v1.0.2")

st.markdown("---")

# --- Load Model and Tools ---
@st.cache_resource
def load_clinical_tools():
    # Initialize the model
    model = DRClassifier()

    # Attempt to load trained weights
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        model_status = "✅ Production Model Loaded"
    except Exception:
        # Fallback: The model already has pretrained EfficientNet weights from the architecture.py
        # We just notify the user it's in "Pre-trained" mode.
        model_status = "⚠️ Using Pre-trained Base Model (Demo Mode)"

    # Target layer for Grad-CAM in EfficientNet-B0
    target_layer = model.base_model.features[-1]
    from src.xai.explainer import DRExplainer
    explainer = DRExplainer(model, target_layer)
    processor = DRImageProcessor()

    return model, explainer, processor, model_status

model, explainer, processor, model_status = load_clinical_tools()

# --- Sidebar: Patient Information ---
st.sidebar.header("📋 Patient Metadata")
patient_id = st.sidebar.text_input("Patient ID", "PAT-2026-001")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
clinic_name = st.sidebar.text_input("Rural Clinic Name", "North Region Health Center")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ AI Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, key="conf_threshold_slider")

st.sidebar.info(f"**Status:** {model_status}")

# --- Main Application ---
uploaded_file = st.file_uploader("📤 Upload Retinal Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Layout: Original vs Explanation
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖼️ Input Scan")
        st.image(image, use_container_width=True, caption=f"Patient: {patient_id}")

    with st.spinner("🔬 AI is analyzing retinal pathology..."):
        # Process and Predict
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_img = processor.preprocess(temp_path)
        input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)

        # Generate XAI Heatmap
        heatmap = explainer.generate_heatmap(input_tensor)
        explanation_img = explainer.apply_heatmap_to_image(heatmap, img_array)

    with col2:
        st.markdown("### 🔍 XAI Explanation")
        st.image(explanation_img, use_container_width=True, caption="Saliency Map (Grad-CAM)")

    # --- Results Section ---
    st.markdown("---")
    st.markdown("### 📋 Diagnostic Report")

    dr_labels = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR"
    }

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("DR Severity", dr_labels[pred.item()])

    with res_col2:
        st.metric("AI Confidence", f"{conf.item()*100:.2f}%")

    with res_col3:
        action = "🚨 Urgent Referral" if pred.item() >= 3 else "✅ Routine Monitoring"
        st.metric("Recommended Action", action)

    # Detailed Clinical Note
    st.markdown(f"""
    <div class="report-box">
        <h4>👨‍⚕️ Clinical Interpretation</h4>
        <p>The AI has detected patterns consistent with <b>{dr_labels[pred.item()]}</b>.
        The red highlighted regions in the saliency map indicate the primary areas of interest (likely exudates or hemorrhages).
        <b>Clinical Action:</b> {action}. Please correlate this finding with a physical examination.</p>
        <small>Generated for Patient {patient_id} at {clinic_name}</small>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please upload a retinal fundus image to generate a diagnostic report.")
    # Display a placeholder or a guide on how to take a good photo
    st.info("💡 **Tip:** Ensure the image is centered and well-lit for the best AI accuracy.")
