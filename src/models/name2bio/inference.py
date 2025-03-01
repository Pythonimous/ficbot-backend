from src.models.name2bio.utils import is_bio_allowed

def generate_bio(name, model, temperature=1.2, min_length=50, max_length=400, *, nsfw_on = False):

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
        token_length = output['usage']['completion_tokens']

        if token_length < min_length:
            continue

        bio = output['choices'][0]['text'].strip()

        if not is_bio_allowed(bio, nsfw_on):
            continue
        
        return f"{first_name} {bio}"


if __name__ == "__main__":
    import os
    import random
    from llama_cpp import Llama

    random_seed = random.randint(0, 2**31 - 1)  # Large random seed

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'files/name2bio.gguf')
    model = Llama(model_path, seed=random_seed)
    print(generate_bio("Kirill Nikolaev", model))