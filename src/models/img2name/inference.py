import argparse
import sys
import os
import pickle

from io import BytesIO
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.models.utils import sample, preprocess_image_array
from src.models.img2name.img2name import Img2Name


def generate_name(model, maps,
                  image_bytes, min_name_length=2, diversity=1.2,
                  start_token="@", end_token="$", ood_token="?"):
    """
    Generates a name using the PyTorch model.

    Args:
        model (nn.Module): The trained Img2Name model.
        maps (tuple): A tuple (char_idx, idx_char) mapping characters to indices and vice versa.
        image_bytes (bytes): Image input in bytes.
        min_name_length (int): Minimum length for the generated name.
        diversity (float): Temperature for sampling.
        start_token (str): Start token for name generation.
        end_token (str): End token for name generation.
        ood_token (str): Out-of-distribution token.

    Returns:
        str: Generated name.
    """
    char_idx, idx_char = maps

    # Load and preprocess the image
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image, dtype=np.float32)
    image_tensor = preprocess_image_array(image_array).unsqueeze(0)  # (1, 3, 224, 224)
    
    device = next(model.parameters()).device  # Get model's device
    image_tensor = image_tensor.to(device)

    maxlen = model.maxlen  # Model’s expected max sequence length

    name_tokens = torch.tensor([[char_idx[start_token]] * maxlen], dtype=torch.long, device=device)  # (1, maxlen)

    generated = ""

    with torch.no_grad():  # No need to track gradients
        while not (generated.endswith(start_token) or generated.endswith(end_token)):
            # Get model predictions
            preds = model(image_tensor, name_tokens)  # (1, vocab_size)
            preds = F.softmax(preds / diversity, dim=-1)  # Apply temperature scaling

            next_char = ood_token
            while next_char == ood_token:  # Resample if OOD token is chosen
                next_index = sample(preds.squeeze().cpu().numpy(), diversity)
                next_char = idx_char[next_index]

            if next_char == end_token and generated.count(' ') < min_name_length - 1:
                next_char = " "  # Ensure minimum name length

            name_tokens = torch.cat([name_tokens[:, 1:], torch.tensor([[char_idx[next_char]]], device=device)], dim=1)
            generated += next_char

    # Remove start/end tokens if present
    if generated[-1] in {start_token, end_token}:
        generated = generated[:-1]

    # Capitalize words
    generated = ' '.join(word.capitalize() for word in generated.split())

    return generated


def parse_arguments():
    parser = argparse.ArgumentParser(prog='Ficbot', description='Your friendly neighborhood fanfic writing assistant! '
                                                                'Boost your imagination with a bit of AI magic.')
    
    parser.add_argument('--info', action='store_true', help='show models available for inference')
    parser.add_argument('--model', nargs='?', help='model to use for inference')
    parser.add_argument('--model_path', nargs='?', help='path to model to infer from')
    parser.add_argument('--img_path', nargs='?', help='path to the input image')
    parser.add_argument('--min_name_length', default=2, type=int, help='minimum length of name')
    parser.add_argument('--diversity', default=1.2, type=float, help='diversity of predictions')

    args = parser.parse_args()
    return args


def load_model(weights_path, parameters_path, model_class):
    """
    Load a PyTorch model from a file.

    Args:
        model_path (str): Path to the model file.
        model_class (type): Model class to instantiate.
        init_params_path (str): Path to the file containing the model's initialization parameters.

    Returns:
        nn.Module: The loaded model.
    """

    with open(parameters_path, "rb") as f:
        init_params = pickle.load(f)

    model = model_class(**init_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_state_dict = torch.load(weights_path, map_location=device)

    if 'model_state_dict' in model_state_dict:  # Checkpoint instead of just a state_dict
        model_state_dict = model_state_dict['model_state_dict']

    model.load_state_dict(model_state_dict)  # Load model for inference

    return model


def main(arguments):
    """
    Main function for running inference.

    Args:
        arguments (argparse.Namespace): Parsed command-line arguments.
    """

    # Display available models if --info is used
    if arguments.info:
        print("Models available for inference:")
        print("img2name: Image to name model")
        print("Good luck!")
        sys.exit()
    
    model_path = arguments.model_path
    weights_path = os.path.join(model_path, f"weights.pt")
    parameters_path = os.path.join(model_path, f"params.pkl")
    maps_path = os.path.join(model_path, f"maps.pkl")

    if not os.path.exists(weights_path):
        raise ValueError(f"Weights path {weights_path} not found")
    if not os.path.exists(parameters_path):
        raise ValueError(f"Model parameters path {parameters_path} not found")
    if not os.path.exists(maps_path):
        raise ValueError(f"Maps path {maps_path} not found")

    # Load image
    image_bytes = open(arguments.img_path, "rb").read()
    
    model = load_model(weights_path, parameters_path, Img2Name)
    
    model.eval()

    # Load character mappings
    with open(maps_path, 'rb') as f:
        maps = pickle.load(f)

    # Generate name
    name = generate_name(model, maps, image_bytes, min_name_length=arguments.min_name_length, diversity=arguments.diversity)
    print(name)

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
    '''
    python src/models/img2name/inference.py --model_path src/models/img2name/files/ --img_path tests/files/sample.jpg --min_name_length 2 --diversity 1.0
    '''