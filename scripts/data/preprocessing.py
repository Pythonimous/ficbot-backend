import os
import pandas as pd

from utils import clear_corpus_characters, clean_bio


def prepare_raw_dataset(df_path, save_path, required_columns = {'name', 'bio', 'image'}):
    mal_characters = pd.read_csv(df_path).fillna('')
    mal_dataset = mal_characters[["eng_name", "bio", "img_index"]].reset_index(drop=True)
    mal_dataset['image'] = mal_dataset['img_index'].map(lambda x: f"{x}.jpg")
    del mal_dataset['img_index']

    mal_dataset['name'] = mal_dataset['eng_name']
    del mal_dataset['eng_name']
    mal_dataset = mal_dataset[['name', 'bio', 'image']]
    if 'name' in required_columns:
        mal_dataset['name'] = clear_corpus_characters(mal_dataset['name'], 100)
    else:
        del mal_dataset['name']
    if 'bio' in required_columns:
        mal_dataset['bio'] = mal_dataset['bio'].map(clean_bio)
    else:
        del mal_dataset['bio']
    if 'image' not in required_columns:
        del mal_dataset['image']
    mal_dataset.to_csv(save_path, index_label=False)


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "anime_characters.csv")
    output_path = os.path.join(current_dir, "name_bio.csv")
    prepare_raw_dataset(input_path, output_path, {'name', 'bio'})


if __name__ == "__main__":
    main()
