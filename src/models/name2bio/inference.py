import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.models.name2bio.utils import is_bio_allowed


def get_similar_characters(query_name, vector_db, k=3):
    query_text = f"Character Name: {query_name}"

    # Retrieve top-k similar characters
    results = vector_db.similarity_search(query_text, k)

    all_genres = set()
    all_themes = set()
    for char in results:
        all_genres.update(char.metadata['anime_genres'].split('|'))
        all_themes.update(char.metadata['anime_themes'].split('|'))
    
    genres = ', '.join(all_genres)
    themes = ', '.join(all_themes)

    return genres, themes


def generate_bio(name, model, vector_db, temperature=1.2, max_length=300, *, nsfw_on = False):

    # Get similar characters

    genres, themes = get_similar_characters(name, vector_db)

    prompt = f"[CHARACTER] {name}\n[GENRES] {genres}\n[THEMES] {themes}\n[BIO]"

    while True:
        output = model.create_completion(
            prompt,
            temperature=temperature,
            max_tokens=max_length,
            repeat_penalty=1.2,
            top_k=50,
            top_p=0.9,
            stop=["[END]"]
        )

        bio = output['choices'][0]['text'].strip()

        if not is_bio_allowed(bio, nsfw_on):
            continue
        
        return bio


def main():

    import random
    import argparse
    from llama_cpp import Llama

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Generate a character bio from a name.")
    
    # Required argument
    parser.add_argument('character_name', type=str, help="The name of the character to generate the bio for.")

    # Optional arguments with default values
    parser.add_argument('--temperature', type=float, default=1.0, help="Temperature for text generation (default: 1.0).")
    parser.add_argument('--max_length', type=int, default=200, help="Maximum length for the bio (default: 200).")

    # Parse arguments
    args = parser.parse_args()

    character_name = args.character_name
    temperature = args.temperature
    max_length = args.max_length

    random_seed = random.randint(0, 2**31 - 1)  # Large random seed

    current_dir = os.path.dirname(os.path.abspath(__file__))

    embed_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings_dir = os.path.join(current_dir, "files/name2bio_embeddings")
    vector_db = FAISS.load_local(embeddings_dir, embed_function, allow_dangerous_deserialization=True)

    model_path = os.path.join(current_dir, "files/name2bio.gguf")
    model = Llama(model_path, seed=random_seed)

    print(generate_bio(character_name, model, vector_db, temperature, max_length))

if __name__ == "__main__":
    main()
