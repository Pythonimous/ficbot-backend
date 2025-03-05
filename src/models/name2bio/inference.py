import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.models.name2bio.utils import is_bio_allowed

def generate_bio(name, model, temperature=1.2, max_length=300, *, nsfw_on = False):

    names = name.split()
    if len(names) > 1:
        first_name = names[0]
    else:
        first_name = f"{name} is"

    prompt = f"[CHARACTER] {name}\n[BIO] {first_name}"

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
        
        return f"{first_name} {bio}"


def main():

    import random
    import argparse
    from llama_cpp import Llama

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
    model_path = os.path.join(current_dir, 'files/name2bio.gguf')
    model = Llama(model_path, seed=random_seed)

    print(generate_bio(character_name, model, temperature, max_length))

if __name__ == "__main__":
    main()
