import os
import requests
import streamlit as st

# Replace 'Your Full Name' with your actual full name if you want it shown in the UI footer.
AUTHOR_NAME = "Your Full Name"

API = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Pneumonia Detection", layout="wide")
st.title("Chest X-Ray Pneumonia Detection")
st.caption("⚠️ Research & education only – Not for clinical use")

# File uploader
uploaded_files = st.file_uploader(
    "Upload X-ray images (PNG, JPG). DICOM not directly supported in this simplified UI.",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

def call_api(file):
    """
    Send file to backend /predict endpoint.
    Returns parsed JSON (or dictionary with 'label' == 'Error').
    """
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        res = requests.post(f"{API}/predict", files=files, timeout=30)
        if res.status_code == 200:
            return res.json()
        else:
            return {"label": "Error", "confidence": 0.0, "gradcam_url": res.text}
    except Exception as e:
        return {"label": "Error", "confidence": 0.0, "gradcam_url": str(e)}

if st.button("Predict", disabled=not uploaded_files):
    for f in uploaded_files:
        result = call_api(f)
        if result.get("label") == "Error":
            st.error(f"{f.name} → Error: {result.get('gradcam_url')}")
        else:
            st.subheader(f"{f.name} → {result['label']} ({result['confidence']:.2%})")
            # Display Grad-CAM from backend static path (backend serves /static/*.png)
            gradcam_path = f"{API}{result['gradcam_url']}"
            try:
                st.image(gradcam_path, caption="Grad-CAM Overlay", use_column_width=True)
            except Exception:
                # Fallback: show message if image can't be loaded from URL
                st.info(f"Grad-CAM saved at: {result['gradcam_url']} (served by backend)")

st.markdown("---")
st.caption(f"👨‍💻 Developed by Nitish Dhamu")
