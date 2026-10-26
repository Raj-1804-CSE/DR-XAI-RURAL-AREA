import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from src.preprocessing.image_processor import DRImageProcessor
from src.model.architecture import DRClassifier
from src.xai.explainer import DRExplainer

# --- Page Configuration ---
st.set_page_config(
    page_title="Rural-DR AI: Explainable Diagnosis",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Rural-DR AI: Explainable Diagnosis")
st.markdown("""
    **Empowering rural healthcare with transparent AI.**
    Upload a retinal fundus image to get an instant DR severity assessment with visual explanations.
""")

# --- Load Model and Tools ---
@st.cache_resource
def load_clinical_tools():
    # Initialize the model
    model = DRClassifier()
    # Load weights if they exist, otherwise use the initialized model for demo
    try:
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        st.success("Trained model loaded successfully!")
    except FileNotFoundError:
        st.warning("Trained model not found. Using an untrained model for UI demonstration.")

    # Target layer for Grad-CAM in EfficientNet-B0 is the last conv layer
    target_layer = model.base_model.features[-1]
    explainer = DRExplainer(model, target_layer)
    processor = DRImageProcessor()

    return model, explainer, processor

model, explainer, processor = load_clinical_tools()

# --- Sidebar Settings ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st.sidebar.info("This tool is designed to assist clinicians, not replace them.")

# --- Main Interface ---
uploaded_file = st.file_uploader("Upload Retinal Fundus Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Original Image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Scan")
        st.image(image, use_container_width=True)

    # 2. Process and Predict
    with st.spinner("Analyzing scan..."):
        # Preprocess image
        # Since processor.preprocess expects a path, we'll adapt it for the uploaded file
        # For simplicity in this UI, we'll save it temporarily or adapt the method
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        processed_img = processor.preprocess(temp_path)
        input_tensor = torch.from_numpy(processed_img).permute(2, 0, 1).float().unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)

        # 3. Generate Explanation
        # We need gradients for XAI, so we run this without torch.no_grad()
        heatmap = explainer.generate_heatmap(input_tensor)
        explanation_img = explainer.apply_heatmap_to_image(heatmap, img_array)

    # DR Severity Labels
    dr_labels = {
        0: "No DR",
        1: "Mild NPDR",
        2: "Moderate NPDR",
        3: "Severe NPDR",
        4: "Proliferative DR"
    }

    with col2:
        st.subheader("AI Explanation (Heatmap)")
        st.image(explanation_img, use_container_width=True)

    # --- Results Section ---
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric("Predicted Severity", dr_labels[pred.item()])

    with res_col2:
        st.metric("Confidence Score", f"{conf.item()*100:.2f}%")

    with res_col3:
        status = "🚨 Urgent Referral" if pred.item() >= 3 else "✅ Regular Monitoring"
        st.metric("Clinical Action", status)

    st.info(f"**Doctor's Note:** The red highlighted areas in the heatmap indicate the primary regions the AI used to determine the diagnosis of **{dr_labels[pred.item()]}**. Please correlate these areas with clinical markers such as hemorrhages or exudates.")

else:
    st.info("Please upload a fundus image to begin the analysis.")
