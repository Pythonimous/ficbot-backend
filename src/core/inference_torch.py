import argparse
import sys
import os
import pickle

from io import BytesIO
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.core.utils import sample, preprocess_image_array
from src.models.img2name.img2name_torch import Img2Name


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
    image_tensor = torch.tensor(preprocess_image_array(image_array), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 224, 224)

    # Move model to evaluation mode
    model.eval()
    
    device = next(model.parameters()).device  # Get model's device
    image_tensor = image_tensor.to(device)

    maxlen = model.maxlen  # Model’s expected max sequence length

    # Start sequence as token indices
    name_tokens = [char_idx[start_token]] * maxlen  # List of indices
    generated = ""

    with torch.no_grad():  # No need to track gradients
        while not (generated.endswith(start_token) or generated.endswith(end_token)):
            # Convert list to tensor (1, maxlen) and send to device
            name_tensor = torch.tensor([name_tokens], dtype=torch.long, device=device)

            # Get model predictions
            preds = model(image_tensor, name_tensor)  # (1, vocab_size)
            preds = F.softmax(preds / diversity, dim=-1)  # Apply temperature scaling

            next_char = ood_token
            while next_char == ood_token:  # Resample if OOD token is chosen
                next_index = sample(preds.squeeze().cpu().numpy(), diversity)
                next_char = idx_char[next_index]

            if next_char == end_token and generated.count(' ') < min_name_length - 1:
                next_char = " "  # Ensure minimum name length

            # Shift tokens: Remove first, add new token
            name_tokens = name_tokens[1:] + [char_idx[next_char]]
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
    parser.add_argument('--maps', nargs='?', help='path to maps for vectorizer (if applicable to model)')
    parser.add_argument('--img_path', nargs='?', help='path to the input image')
    parser.add_argument('--min_name_length', default=2, type=int, help='minimum length of name')
    parser.add_argument('--diversity', default=1.2, type=float, help='diversity of predictions')

    args = parser.parse_args()
    return args


def main(arguments):
    """
    Main function for running inference.

    Args:
        arguments (argparse.Namespace): Parsed command-line arguments.
    """

    # Display available models if --info is used
    if arguments.info:
        print("Models available for inference:")
        print("simple_img_name: Image to name model")
        print("Good luck!")
        sys.exit()

    # Load image
    image_bytes = open(arguments.img_path, "rb").read()

    models_dict = {'simple_img_name': Img2Name}

    if arguments.model not in models_dict:
        raise ValueError("Model not found")
    
    init_params_path = os.path.join(os.path.dirname(arguments.model_path), "init_params.pkl")
    if os.path.exists(init_params_path):
        with open(init_params_path, "rb") as f:
            init_params = pickle.load(f)
    else:
        raise ValueError("Model parameters not found")
    
    weights_path = os.path.join(os.path.dirname(arguments.model_path), "model.pt")
    if not os.path.exists(weights_path):
        raise ValueError("Model weights not found")
    
    model = models_dict[arguments.model](**init_params)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))  # Load model for inference

    # Load character mappings
    if arguments.maps:
        with open(arguments.maps, 'rb') as f:
            maps = pickle.load(f)
    else:
        raise ValueError("Maps path not provided")
    # Generate name
    name = generate_name(model, maps, image_bytes, min_name_length=arguments.min_name_length, diversity=arguments.diversity)
    print(name)

if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
