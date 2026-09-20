import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewFace AI App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Heavy Model / Resource Loader (Cached to prevent startup timeouts)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading face detection models...")
def load_face_detection_model():
    """
    Loads OpenCV's Haar Cascade classifier or your custom face detection model.
    Caching ensures this code runs only once when the server boots.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("Failed to load face detection cascade XML.")
        return None
    return face_cascade

# Load model at runtime safely
face_cascade = load_face_detection_model()

# -----------------------------------------------------------------------------
# Processing Functions
# -----------------------------------------------------------------------------
def detect_faces(image: Image.Image, scale_factor: float, min_neighbors: int):
    """
    Detects faces in an RGB Image and draws bounding boxes around them.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    if face_cascade is None:
        return img_array, []
        
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )
    
    annotated_img = img_array.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
    return annotated_img, faces

# -----------------------------------------------------------------------------
# Application Layout & UI
# -----------------------------------------------------------------------------
def main():
    st.title("👤 NewFace - Facial Recognition & Analysis")
    st.markdown(
        "Welcome to the **NewFace** application. Upload an image or capture one from your webcam "
        "to perform real-time face detection and processing."
    )

    # Sidebar Controls
    st.sidebar.header("Settings & Parameters")
    
    input_source = st.sidebar.radio(
        "Choose Input Source",
        ["Image Upload", "Webcam Capture"]
    )

    scale_factor = st.sidebar.slider(
        "Scale Factor",
        min_value=1.05,
        max_value=1.50,
        value=1.10,
        step=0.05,
        help="Parameter specifying how much the image size is reduced at each image scale."
    )

    min_neighbors = st.sidebar.slider(
        "Min Neighbors",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Parameter specifying how many neighbors each candidate rectangle should have to retain it."
    )

    image_file = None

    if input_source == "Image Upload":
        image_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp"]
        )
    else:
        image_file = st.camera_input("Take a picture")

    # Main Processing Section
    if image_file is not None:
        try:
            image = Image.open(image_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Detection Result")
                with st.spinner("Processing image..."):
                    result_img, detected_faces = detect_faces(
                        image, scale_factor, min_neighbors
                    )
                    st.image(result_img, use_column_width=True)

            # Display Stats
            st.divider()
            num_faces = len(detected_faces)
            if num_faces > 0:
                st.success(f"Detected **{num_faces}** face(s) in the image.")
            else:
                st.warning("No faces detected with current sensitivity settings.")

        except Exception as e:
            st.error(f"An error occurred while processing the image: {str(e)}")
    else:
        st.info("Please upload an image or capture one using your webcam to get started.")

if __name__ == "__main__":
    main()