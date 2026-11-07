
# libs
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# sanity check module
from sanity_check import SanityCheck

# page configuration
st.set_page_config(page_title="Plant Disease Classifier")
st.title("WELCOME TO PLANT DISEASE CLASSIFIER.")
st.caption("AI-powered plant leaf disease detection for tomato and bell pepper")
st.markdown("""
**Leaf Doctor** looks at a photo of a plant leaf and tells you whether it is healthy or which
disease it shows. It covers **12 conditions** across **tomato** and **bell pepper** leaves.

#### How to use it
1. Take a clear photo of **one leaf** (or upload one).
2. The app first checks that the image really contains a leaf.
3. If it does, the disease model predicts the condition and shows how confident it is.

#### How it works
- **Step 1, leaf check:** a pre-trained vision-language model (CLIP) rejects non-leaf images such as hands, objects and walls.
- **Step 2, disease classifier:** a fine-tuned MobileNetV2 deep-learning model trained on thousands of leaf images.
- **Confidence:** if the model is not sure, it says **Uncertain** rather than guessing.

#### Tips for a good photo
- Fill the frame with a single leaf and keep it in focus.
- Use natural daylight and avoid strong shadows or glare.
- Use a plain background if possible.

##### Please note:
*This tool is a **support aid for learning and early screening**. It can make mistakes, especially
with blurry photos, unusual lighting, or leaves from other plants. For a final diagnosis or
treatment advice, consult an agricultural expert.*
""")

# models
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(
        "plant_disease_model.keras",
        compile=False
    )

model = load_my_model()

class_names = ["Pepper__bell___Bacterial_spot", "Pepper__bell___healthy", 
               "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
               "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites_Two_spotted_spider_mite",
               "Tomato__Target_Spot", "Tomato__Tomato_YellowLeaf__Curl_Virus",
               "Tomato__Tomato_mosaic_virus", "Tomato_healthy"]
icons = ["🦠", "🫑", "🦠", "🍂", "🥀", "🍄", "🔴", "🕷️", "🎯", "🌀", "🧩","🍅"]                      # Tomato: healthy

# inputs
source = st.radio("Choose Input:", ["Upload Image", "Use Camera"], horizontal=True)

if source == "Upload Image":
    camera_image = st.file_uploader("Upload a leaf photo", type=['jpg', 'jpeg', 'png'])
else:
    camera_image = st.camera_input("Capture clear leaf image:- ", resolution='1080p')

if camera_image is not None:

    # IMAGE PROCESSING
    image = Image.open(camera_image).convert('RGB')
    st.image(image, caption='your uploaded object!!!')

    image = image.resize((224, 224))
    image_arr  = np.array(image, dtype=np.float64)
    image_arr = np.expand_dims(image_arr, axis=0)

    # sanity check [HUGGING FACE]
    check = SanityCheck(image)
    st.subheader("Sanity Check: ")
    
    if check == 0:
        st.success("[PASSED] Leaf image detected!")

        # main model prediction [MOBILENETV2]
        pred = model.predict(image_arr, verbose=0).ravel()
        scores = pred[0]
        
        # results
        st.subheader("Model Prediction: ")
        
        # decode
        k = int(np.argmax(pred))      # position of the largest number
        
        st.info(f"{icons[k]} {class_names[k]}")
        
    else:
        st.error("[BLOCKED] Please provide leaf image!")
