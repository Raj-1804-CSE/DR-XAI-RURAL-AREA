import sys
import os

# Add the root directory to the Python path
# This ensures that 'from src...' imports work regardless of where the script is run from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import the app logic
try:
    import src.ui.app as app
    print("Successfully launched Rural-DR AI from main.py")
except ImportError as e:
    print(f"Import Error: {e}")
    # Fallback for different environment structures
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    import ui.app as app
