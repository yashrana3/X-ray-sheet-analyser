from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Tuberculosis Analyzer API")

# Setup CORS to allow requests from the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    with open("model/model.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("model/model_weights.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Target classes as defined in the original tfjs app
TARGET_CLASSES = {
    0: 'Normal',
    1: 'Tuberculosis'
}

@app.get("/")
def read_root():
    return {"status": "Tuberculosis Analyzer API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded"})

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess the image exactly like the tfjs app
        # .resizeNearestNeighbor([96,96])
        image = image.resize((96, 96), Image.NEAREST)
        
        # Convert to numpy array and ensure it's float32
        img_array = np.array(image, dtype=np.float32)
        
        # .toFloat().div(tf.scalar(255.0))
        img_array = img_array / 255.0
        
        # .expandDims()
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        prob_tb = float(prediction[0][0])
        prob_normal = 1.0 - prob_tb
        
        # Format the predictions
        results = [
            {"className": "Tuberculosis", "probability": prob_tb},
            {"className": "Normal", "probability": prob_normal}
        ]
            
        # Sort exactly as in the frontend
        results = sorted(results, key=lambda x: x["probability"], reverse=True)
        
        return {"predictions": results}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
