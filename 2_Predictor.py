import streamlit as st
import numpy as np
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import nibabel as nib
import base64
import io
import tempfile
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocess_and_loader import file_loader, Preprocess, PretrainedUNet2D, Predictor

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: scroll;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background
set_bg_from_local("background_2.jpg")

st.title("Brain Tumor Segmentation - Glioma Tumors")

required_keys = ['flair_file', 't1_file', 't1ce_file', 't2_file']
if not all(k in st.session_state for k in required_keys):
    st.warning("Please upload all files from the first page.")
    st.stop()

flair_file = st.session_state['flair_file']
t1_file = st.session_state['t1_file']
t1ce_file = st.session_state['t1ce_file']
t2_file = st.session_state['t2_file']

# Initialize session state variables
if 'pred_ready' not in st.session_state:
    st.session_state.pred_ready = False
if 'pred_bytes' not in st.session_state:
    st.session_state.pred_bytes = None
if 'prediction_running' not in st.session_state:
    st.session_state.prediction_running = False

def run_prediction():
    st.session_state.prediction_running = True
    
    progress = st.progress(0, text="Starting prediction...")

    progress.progress(10, "Writing temporary files...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_flair, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_t1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_t1ce, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_t2:

            tmp_flair.write(flair_file.read())
            tmp_t1.write(t1_file.read())
            tmp_t1ce.write(t1ce_file.read())
            tmp_t2.write(t2_file.read())

    progress.progress(25, "Loading and preprocessing images...")
    flair_data, t1_data, t1ce_data, t2_data = file_loader(tmp_flair.name, tmp_t1.name, tmp_t1ce.name, tmp_t2.name)
    input_slices = Preprocess(flair_data, t1_data, t1ce_data, t2_data)

    progress.progress(45, "Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load('unet2d.pth', map_location=device)
    model = PretrainedUNet2D(in_channels=4, num_classes=4, encoder_name='resnet34', pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    progress.progress(65, "Running prediction...")
    Predictions = Predictor(input_slices, model, device)
    pred = np.stack(Predictions, axis=-1)
    pred = pred.astype(np.int16)

    flair_nifti = nib.load(tmp_flair.name)
    affine = flair_nifti.affine
    pred_nifti = nib.Nifti1Image(pred, affine)

    progress.progress(85, "Saving prediction to NIfTI file...")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp_pred:
        nib.save(pred_nifti, tmp_pred.name)
        tmp_pred_path = tmp_pred.name

    with open(tmp_pred_path, "rb") as f:
        file_bytes = f.read()

    st.session_state.pred_bytes = file_bytes
    st.session_state.pred_ready = True
    st.session_state.prediction_running = False

    progress.progress(100, "Prediction completed successfully!")
    
    # Clean up temporary files
    try:
        os.unlink(tmp_flair.name)
        os.unlink(tmp_t1.name)
        os.unlink(tmp_t1ce.name)
        os.unlink(tmp_t2.name)
        os.unlink(tmp_pred_path)
    except:
        pass  # Ignore cleanup errors

# --- Main UI Logic ---
if not st.session_state.pred_ready and not st.session_state.prediction_running:
    # Show prediction button when no prediction is ready and not currently running
    if st.button("Run Prediction", type="primary"):
        run_prediction()
        st.rerun()  # Refresh to show the download button

elif st.session_state.prediction_running:
    # Show that prediction is running
    st.info("Prediction is running... Please wait.")
    st.button("Run Prediction", disabled=True)

elif st.session_state.pred_ready:
    # Show both download button and option to run new prediction
    st.success("✅ Prediction completed! You can download the results below.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.download_button(
            label="📥 Download NIfTI Prediction",
            data=st.session_state.pred_bytes,
            file_name="prediction.nii.gz",
            mime="application/gzip",
            type="primary",
            help="Click to download the segmentation results as a NIfTI file"
        )
    
    with col2:
        if st.button("🔄 Run New Prediction", help="Run prediction again with the new files"):
            # Reset the prediction state to allow running again
            st.session_state.pred_ready = False
            st.session_state.pred_bytes = None
            st.switch_page("pages/1_Uploader.py")

# Display file information
with st.expander("📁 Uploaded Files Information"):
    st.write(f"**FLAIR:** {flair_file.name}")
    st.write(f"**T1:** {t1_file.name}")
    st.write(f"**T1CE:** {t1ce_file.name}")
    st.write(f"**T2:** {t2_file.name}")