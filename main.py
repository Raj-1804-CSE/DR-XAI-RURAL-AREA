import streamlit as st
import sys
import os

# 1. Fix the Python Path immediately
# This allows the app to find the 'src' folder regardless of the deployment environment
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. Now import the UI logic
# We wrap this in a try-except to give a clear error message on the screen if it fails
try:
    from src.ui.app import load_clinical_tools

    # We need to execute the code inside src/ui/app.py
    # Since src/ui/app.py is written as a script, we can use exec()
    # or simply import the module if the logic is inside a function.
    # To be safe and ensure the UI renders, we execute the app file.

    with open("src/ui/app.py", "r") as f:
        code = f.read()
        exec(code, globals())

except Exception as e:
    st.error("🚨 Application Error")
    st.exception(e)
    st.info("Please check the logs in 'Manage App' for more details.")
