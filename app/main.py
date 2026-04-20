import os
import numpy as np
import json 
from PIL import Image
import requests
# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import streamlit as st

API_URL = "https://horticulture-helper-production-d868.up.railway.app"

working_dir = os.path.dirname(os.path.abspath(__file__))
# model_path = "app/trained_model/fruits_disease_prediction_model_updated.h5"
model_path = os.path.join(working_dir, "trained_model", "fruits_disease_prediction_model_updated.h5")
model = tf.keras.models.load_model(model_path)
# model = load_model(model_path)
print("Model loaded successfully.")

class_indices_mapping=json.load(open(f"{working_dir}/class_indices.json")) #loading the index class names mappingn 

def input_img_preprocessing(input_img, target_size=(224, 224)):
    input_img.seek(0)
    img=Image.open(input_img).convert("RGB")
    img=img.resize(target_size)
    img_arr=np.array(img)
    img_arr=np.expand_dims(img_arr, axis=0)
    img_arr=img_arr.astype('float32')/255. #rescaling
    return img_arr
 
def predict_img_class(model, input_img, class_indices):
    preprocessed_img=input_img_preprocessing(input_img)
    prediction=model.predict(preprocessed_img)
    prediction_index=np.argmax(prediction, axis=1)[0]
    prediction_class=class_indices[str(prediction_index)]
    return prediction_class

# Frontend
st.title("🌿 Plant Disease Classifier")
uploaded_img=st.file_uploader("Upload an Image...",type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image=uploaded_img
    coln1, coln2= st.columns(2)
    
    with coln1:
        uploaded_img.seek(0)
        st.image(Image.open(uploaded_img))

    with coln2:
        if st.button("Classify"):
            uploaded_img.seek(0)
            response = requests.post(
               f"{API_URL}/predict",
                files={"file": uploaded_img}
            )

            st.session_state.result = response.json()

            st.success(f"Prediction: {st.session_state.result['prediction']}")
            st.info(f"Confidence: {st.session_state.result['confidence']:.2f}")

    st.subheader("Treatment Advice 🌱", divider="green")
    # st.write("INFO...")
    if 'result' in st.session_state:
        response = requests.get(
            f"{API_URL}//treatment",
            params={
                "disease": st.session_state.result['prediction'],
                "confidence_score": st.session_state.result['confidence']
            }
            
        )
        if response.status_code == 200:
            data=response.json()
            # st.write(data) #debug

            st.subheader("🌱 Disease Confirmation")
            st.write(data["disease_confirmation"])

            st.subheader("🦠 Description")
            st.write(data["description"])

            tab1, tab2, tab3= st.tabs(["💊 Organic Treatment", "⚗️ Chemical Treatment", "🛡️ Prevention"])

        
            with tab1:
                for item in data["organic_treatment"]:
                    st.write("✅", item)

            with tab2:
                for item in data["chemical_treatment"]:
                    st.write("🧪", item)

            with tab3:
                for item in data["prevention"]:
                    st.write("🌿", item)

            st.warning(data["note"])
        else:
            print("Error:", response.status_code)
            print(response.text)
        