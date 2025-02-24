import argparse
import sys
import os
import pickle

from io import BytesIO
from PIL import Image

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.core.utils import sample, preprocess_image_array


def generate_name(model, maps,
                  image_bytes, min_name_length=2, diversity=1.2,
                  start_token="@", end_token="$", ood_token="?"):
    """ Generates a name using the model inside the container. """

    char_idx, idx_char = maps

    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image, dtype=np.float32)
    image_features = np.expand_dims(preprocess_image_array(image_array), axis=0)

    maxlen = model.get_layer("NAME_INPUT").output.shape[1]

    generated = ""
    name = start_token * maxlen

    while not (generated.endswith(start_token) or generated.endswith(end_token)):
        x_pred_text = np.zeros((1, maxlen, len(idx_char)))
        for t, char in enumerate(name):
            x_pred_text[0, t, char_idx[char]] = 1.0

        preds = model.predict([image_features, x_pred_text], verbose=0)[0]
        next_char = ood_token
        while next_char == ood_token:  # in case next_char is ood token, we sample (and then resample) until it isn't
            next_index = sample(preds, diversity)
            next_char = idx_char[next_index]
        if next_char == end_token and generated.count(' ') < min_name_length - 1:
            next_char = " "

        name = name[1:] + next_char
        generated += next_char

    if generated[-1] in {start_token, end_token}:
        generated = generated[:-1]

    generated = [word.capitalize() for word in generated.split()]
    generated = ' '.join(generated)
    return generated


def parse_arguments():
    parser = argparse.ArgumentParser(prog='Ficbot', description='Your friendly neighborhood fanfic writing assistant! '
                                                                'Boost your imagination with a bit of AI magic.')
    
    parser.add_argument('--info', action='store_true', help='show models available for inference')
    parser.add_argument('--model', default='simple_img_name', choices=['simple_img_name'],
                        help='the model you want to train')
    parser.add_argument('--model_path', nargs='?', help='path to model to infer from')
    parser.add_argument('--maps', nargs='?', help='path to maps for vectorizer (if applicable to model)')
    parser.add_argument('--img_path', nargs='?', help='path to the input image')
    parser.add_argument('--min_name_length', default=2, type=int, help='minimum length of name')
    parser.add_argument('--diversity', default=1.2, type=float, help='diversity of predictions')

    args = parser.parse_args()
    return args


def main(arguments):

    if arguments.info:
        print("Models available for inference:")
        print("simple_img_name: Image to name model")
        print("Good luck!")
        sys.exit()

    image_bytes = open(arguments.img_path, "rb").read()

    if arguments.model_path:
        model = tf.keras.models.load_model(arguments.model_path)
    else:
        raise ValueError("Model path not provided")
    
    if arguments.maps:
        with open(arguments.maps, 'rb') as f:
            maps = pickle.load(f)
    else:
        raise ValueError("Maps path not provided")
    
    name = generate_name(model, maps, image_bytes, min_name_length=arguments.min_name_length, diversity=arguments.diversity)
    
    print(name)

if __name__ == "__main__":
    
    arguments = parse_arguments()

    main(arguments)
