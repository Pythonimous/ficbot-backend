import tensorflow as tf
import base64
import pickle
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.api.config import settings, MODEL_PATH, MAPS_PATH
from src.inference import generate_name

# Initialize FastAPI app
app = FastAPI(title="Ficbot Model Inference", version="1.0")

if settings.testing:
    MODEL_PATH = "src/models/img2name/files/img2name.keras"
    MAPS_PATH = "src/models/img2name/files/maps.pkl"

# Load TensorFlow model on startup
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Load character mappings
with open(MAPS_PATH, "rb") as mp:
    maps = pickle.load(mp)

print("Model and mappings loaded!")

@app.post("/generate/")
async def generate_character_name(request: Request):
    """ Receives an image, runs inference, and returns a generated name. """
    try:
        body = await request.json()
        image_bytes = base64.b64decode(body["image"])
        diversity = float(body.get("diversity", 1.2))
        min_name_length = int(body.get("min_name_length", 2))

        name = generate_name(model, maps, image_bytes, min_name_length=min_name_length, diversity=diversity)

        return JSONResponse(content={"success": True, "name": name})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """ Health check endpoint to confirm the API is running. """
    return JSONResponse(content={"status": "OK", "message": "API is running!"})