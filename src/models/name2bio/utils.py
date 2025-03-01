from src.models.name2bio.config import STOP_WORDS, STOP_WORDS_NSFW, STOP_PHRASES, STOP_PHRASES_NSFW

def is_bio_allowed(bio, nsfw_on = False):
    bio_words = ["".join([c for c in word if c.isalpha()]) for word in bio.lower().split()]
    if set(bio_words).intersection(STOP_WORDS):
        return False
    for phrase in STOP_PHRASES:
        if phrase in bio:
            return False
    if not nsfw_on:
        if set(bio_words).intersection(STOP_WORDS_NSFW):
            return False
        for phrase in STOP_PHRASES_NSFW:
            if phrase in bio:
                return False
    return True

    