from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import io
from app.main import  model, class_indices_mapping
from app.llm_service import get_llm_response

app = FastAPI()

def preprocess_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_arr = np.array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = img_arr.astype('float32') / 255.0
    return img_arr

@app.get("/")
def home():
    return {"message": "Plant Disease Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    
    processed_img = preprocess_image(image_bytes)
    prediction = model.predict(processed_img)
    
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_indices_mapping[str(predicted_index)]
    confidence = float(np.max(prediction))

    return {
        "prediction": predicted_class,
        "confidence": confidence
    }

@app.get("/treatment")
async def getTreatment(disease: str, confidence_score: float):
    result=get_llm_response(disease, confidence_score)
    return result

