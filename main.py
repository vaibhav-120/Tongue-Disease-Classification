import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="", layout="wide")

model = load_model("model2.keras")
model.summary()


CLASS_NAMES = ["Diabetes","Healthy"]

st.title("Tongue Diabetes Classification")
st.write("Upload an image of tongue")

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image = image.resize((640, 640))
    image_array = np.expand_dims(image, axis=0)

    predictions = model.predict(image_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.subheader("Prediction")
    st.success(f"**Prediction** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")
