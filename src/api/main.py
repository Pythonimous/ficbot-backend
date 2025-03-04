import base64
import pickle
import random
from io import BytesIO
from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

import torch
from PIL import Image
from llama_cpp import Llama

from src.api.config import settings, IMG2NAME_WEIGHTS_PATH, IMG2NAME_MAPS_PATH, IMG2NAME_PARAMETERS_PATH, NAME2BIO_MODEL_PATH
from src.models.img2name.img2name import Img2Name
from src.models.img2name.inference import generate_name
from src.models.name2bio.inference import generate_bio

# Initialize FastAPI app
app = FastAPI(title="Ficbot Model Inference", version="1.1")

if settings.testing:
    IMG2NAME_WEIGHTS_PATH = "src/models/img2name/files/weights.pt"
    IMG2NAME_MAPS_PATH = "src/models/img2name/files/maps.pkl"
    IMG2NAME_PARAMETERS_PATH = "src/models/img2name/files/params.pkl"

    NAME2BIO_MODEL_PATH = "src/models/name2bio/files/name2bio.gguf"

# Load Torch model on startup
print("Loading Img2Name model...")
img2name_model = Img2Name.load_model(IMG2NAME_WEIGHTS_PATH, IMG2NAME_PARAMETERS_PATH)
img2name_model.eval()

# Load character mappings
with open(IMG2NAME_MAPS_PATH, "rb") as mp:
    img2name_maps = pickle.load(mp)

print("Loading Name2Bio model...")
random_seed = random.randint(0, 10**10)
name2bio_model = Llama(NAME2BIO_MODEL_PATH, seed=random_seed)

print("Loading AnimeGAN2 model...")
img2anime_model = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
img2anime = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

print("Models and mappings loaded!")

@app.post("/generate")
async def generate_character(request: Request):
    """ Generates a character name from an image, or bio from a name. """
    body = await request.json()

    # Determine the generation type (name or bio)
    generate_type = body.get("type", "name")  # Default to 'name'

    # Determine input type (image or name)
    input_image = body.get("image")
    input_name = body.get("name")

    if not input_image and not input_name:
        raise HTTPException(status_code=400, detail="Either 'image' or 'name' must be provided.")

    diversity = float(body.get("diversity", 1.0))

    if generate_type == "name":
        if not input_image:
            raise HTTPException(status_code=400, detail="Image must be provided for name generation.")
        min_name_length = int(body.get("min_name_length", 2))
        image_bytes = base64.b64decode(input_image)
        result = generate_name(img2name_model, img2name_maps, image_bytes, min_name_length=min_name_length, diversity=diversity)
        return JSONResponse(content={"success": True, "name": result})
    
    if generate_type == "bio":
        if not input_name:
            raise HTTPException(status_code=400, detail="Name must be provided for bio generation.")
        max_bio_length = int(body.get("max_bio_length", 200))
        nsfw_on = body.get("nsfw_on", False)
        result = generate_bio(input_name, name2bio_model, diversity, max_length=max_bio_length, nsfw_on=nsfw_on)
        return JSONResponse(content={"success": True, "bio": result})


@app.post("/convert_to_anime")
async def convert_to_anime(request: Request):
    """ Converts a given image into an anime-stylized image using AnimeGAN2. """
    body = await request.json()
    input_image = body.get("image")

    if not input_image:
        raise HTTPException(status_code=400, detail="Image must be provided for anime conversion.")
    
    try:
        image_bytes = base64.b64decode(input_image)
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Apply anime conversion
        out = img2anime(img2anime_model, img)
        
        # Convert output image to base64
        buffer = BytesIO()
        out.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return JSONResponse(content={"success": True, "anime_image": encoded_image})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.get("/health")
async def health_check():
    """ Health check endpoint to confirm the API is running. """
    return JSONResponse(content={"status": "OK", "message": "API is running!"})