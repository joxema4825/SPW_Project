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
st.markdown("""
            This web app uses a 2D U-Net model trained on BraTS competition dataset containing MRI scans of brain of individuals affected with Glioma Tumor to segment parts of Brain. Please upload a Brain MRI scan to see segmentation of tumor.
            """)

st.markdown("""
            You have to upload 4 input files, T1-Weighted MRI, T2-Weighted MRI, T1-Weighted MRI with constant enhancement, FLAIR. Please upload files of the same brain.
            """)

if st.button("Click here to continue"):
    st.switch_page("pages/1_Uploader.py")