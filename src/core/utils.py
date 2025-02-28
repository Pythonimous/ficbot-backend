import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

def sample(preds, temperature=1.0):
    """ Helper function to sample an index from a probability array """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def preprocess_image_array(image_array, target_size=(224, 224), preprocess_for="mobilenet"):
    """
    Preprocesses an image array for a specified model.
    
    Args:
        image_array (numpy.ndarray): The input image array (H, W, C).
        target_size (tuple): The target image size (height, width). If (H, W, C) is passed, ignores C.
        preprocess_for (str): Name of the model (mobilenet, resnet, etc.).
    
    Returns:
        torch.Tensor: Preprocessed image tensor (C, H, W).
    """

    transform_pipeline = T.Compose([
        T.ToPILImage(),  # Convert numpy array to PIL image
        T.Resize(target_size),  # ✅ Now (H, W) is correctly passed
        T.ToTensor(),  # Convert to (C, H, W) tensor
    ])
    
    image_tensor = transform_pipeline(image_array)

    # Normalize based on the chosen model
    normalize_dict = {
        "mobilenet": T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # MobileNet [-1, 1]
        "resnet": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet, VGG
        "vgg": T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # VGG
    }

    normalize = normalize_dict.get(preprocess_for.lower(), normalize_dict["mobilenet"])  # Default to MobileNet

    return normalize(image_tensor)


def get_image(path, target_size=(224, 224), preprocess_for="mobilenet"):
    image = Image.open(path).convert("RGB")  # Ensure RGB mode
    image_array = np.array(image, dtype=np.uint8)  # Convert to numpy array
    return preprocess_image_array(image_array, target_size, preprocess_for)
