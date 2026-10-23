
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Plant Disease Detector")
st.title("Welcome to plant disease detector.")
st.write("Capture an image and know whether it is healthy or diseased?")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "plant_disease_model.keras",
        compile=False
    )

model = load_my_model()

class_names = ["Diseased", "Healthy"]

# camera input
camera_image = st.camera_input("Capture leaf image:- ", resolution='1080p')

if camera_image is not None:
    image = Image.open(camera_image).convert('RGB')
    st.image(image, caption='your uploaded leaf!!!')

    image = image.resize((224, 224))

    image_arr  = np.array(image, dtype=np.float64)
    image_arr = np.expand_dims(image_arr, axis=0)

    image_arr = tf.keras.applications.mobilenet_v2.preprocess_input(image_arr)

    pred = model.predict(image_arr, verbose=0).ravel()
    scores = pred[0]

    st.subheader("Model Prediction: ")

    # intepret
    THRESHOLD = 0.5
    prob = (scores > THRESHOLD).astype(int)
    
    st.info(f"Leaf is: {class_names[prob]}")
