import streamlit as st
import base64

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

col1, col2 = st.columns(2)

with col1:
    flair_file = st.file_uploader("Upload FLAIR MRI", type=["nii", "nii.gz"])
    t1ce_file = st.file_uploader("Upload T1-Weighted with constant enhancement MRI scan", type=["nii", "nii.gz"])

with col2:
    t1_file = st.file_uploader("Upload T1-Weighted MRI scan", type=["nii", "nii.gz"])
    t2_file = st.file_uploader("Upload T2-Weighted MRI scan", type=["nii", "nii.gz"])
  

if all([flair_file, t1_file, t1ce_file, t2_file]):
    st.session_state['flair_file'] = flair_file
    st.session_state['t1_file'] = t1_file
    st.session_state['t1ce_file'] = t1ce_file
    st.session_state['t2_file'] = t2_file
    st.success("All files uploaded. Redirecting to prediction page...")

    # Navigate to the second page
    st.switch_page("pages/2_Predictor.py")

else:
    st.warning("Please upload all 4 files.")