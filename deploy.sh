#!/bin/bash

echo "🚀 Starting Deployment Process for Rural-DR AI..."

# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Verify model exists
if [ ! -f "models/best_model.pth" ]; then
    echo "⚠️ Warning: models/best_model.pth not found. The app will run in Demo Mode."
else
    echo "✅ Trained model found. Ready for production."
fi

# 4. Launch the app
echo "🌐 Launching Streamlit app on http://localhost:8501"
streamlit run src/ui/app.py
