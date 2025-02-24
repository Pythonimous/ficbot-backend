import numpy as np

from io import BytesIO
from PIL import Image

from src.utils import sample, preprocess_image_array

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