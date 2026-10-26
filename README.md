# Diabetic Retinopathy Explainable AI (XAI)
Project for diagnosing Diabetic Retinopathy in rural areas with explainable AI insights.

## Project Structure
- `data/`: Raw and processed retinal fundus images.
- `models/`: Saved model weights and configurations.
- `notebooks/`: Exploratory Data Analysis (EDA) and experimentation.
- `src/`:
    - `preprocessing/`: Image cleaning, normalization, and augmentation.
    - `model/`: Architecture definition and training loops.
    - `xai/`: Implementation of explainability methods (e.g., Grad-CAM).
    - `ui/`: The user interface for clinicians.
- `tests/`: Unit tests for data and model pipelines.
