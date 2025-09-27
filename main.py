import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from rembg import remove, new_session
from PIL import Image
import face_detect

st.set_page_config(page_title="Tongue Diabetes Classification", layout="wide")

model = load_model("model.keras")
segmentation_model = "u2net.onnx"
CLASS_NAMES = ["Diabetes", "Healthy"]

def remove_background(input_image, model_path):
    session = new_session(model_name="u2net", model_path=model_path)
    
    output_image = remove(input_image, session=session)
    
    black_bg = Image.new('RGBA', output_image.size, (0, 0, 0, 255))
    black_bg.paste(output_image, (0, 0), output_image)
    
    black_bg = black_bg.convert('RGB')
    return black_bg

st.title("Tongue Diabetes Classification")
st.write("Upload an image **or** take a picture of the tongue")

# ---- Session state keys ----
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "captured_image" not in st.session_state:
    st.session_state.captured_image = None
if "confirmed_image" not in st.session_state:
    st.session_state.confirmed_image = None


uploaded = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])
if uploaded:
    st.session_state.confirmed_image = uploaded
    st.session_state.show_camera = False
    st.session_state.captured_image = None
st.markdown("OR")
if st.button("📷 Use Camera"):
    st.session_state.show_camera = True
    st.session_state.captured_image = None
    st.session_state.confirmed_image = None

if st.session_state.show_camera:
    img = st.camera_input("Take a picture (click Use photo to capture)")
    if img is not None:
        st.session_state.captured_image = img
        st.session_state.show_camera = False   
        st.rerun()               

if st.session_state.captured_image and not st.session_state.confirmed_image:
    st.image(st.session_state.captured_image, caption="Captured Image", use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Retake"):
            st.session_state.captured_image = None
            st.session_state.show_camera = True
            st.rerun()
    with c2:
        if st.button("✅ Use This One"):
            st.session_state.confirmed_image = st.session_state.captured_image
            st.session_state.captured_image = None
            st.rerun()

# ---- Prediction after confirmation ----
if st.session_state.confirmed_image:
    image = Image.open(st.session_state.confirmed_image).convert("RGB")
    cropped_image = face_detect.crop_nose_to_chin(image)
    segmented_image = remove_background(cropped_image, segmentation_model)
    org_img_col, crop_img_col, seg_img_col = st.columns(3)
    with org_img_col:
        with st.container():
            st.image(image, caption="Selected Image for Prediction", use_container_width=True)
    with crop_img_col:
        with st.container():
            st.image(cropped_image, caption="Cropped Image (Tongue)", use_container_width=True)
    with seg_img_col:
        with st.container():
            st.image(segmented_image, caption="Image after Background Removal", use_container_width=True)

    img_resized = image.resize((640, 640))
    img_array = np.expand_dims(img_resized, axis=0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction")
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    if st.button("📸 Use with next image"):
        for key in ["captured_image", "confirmed_image", "show_camera"]:
            st.session_state.pop(key, None)
        st.rerun()
