import re
import unicodedata

from collections import defaultdict

from num2words import num2words


def replace_text_numbers(text):
    """ Replaces all numbers in a string with words """
    if not text.endswith(' '):
        text += ' '  # a crutch for end tokens, to be fixed someday
    num_from = 0
    num_to = 0
    current_number = ""
    for i in range(len(text)):
        if current_number == "0" and not text[num_from + 1] == ' ':  # edge case: 02
            text = text[:num_from + 1] + ' ' + text[num_to + 1:]
            return replace_text_numbers(text)
        elif text[i].isnumeric() or (text[i] == "." and current_number and "." not in current_number):
            if not current_number:
                num_from = i
            current_number += text[i]
            num_to = i

        elif current_number:
            mode = 'cardinal'
            if len(current_number) > 1 and current_number.endswith("."):
                current_number = current_number[:-1]
            elif text[num_to + 1: num_to + 3] in ['st', 'nd', 'rd', 'th']:  # edge case: the 7th
                mode = 'ordinal'
                text = text[: num_to] + text[num_to + 3:]

            if mode == 'ordinal' and '.' in current_number:  # edge case: the 7.5th
                current_number = current_number.split('.')
                current_number = num2words(float(current_number[0])) \
                                 + " point " \
                                 + num2words(float(current_number[1]), to=mode)
            else:
                current_number = num2words(float(current_number), to=mode)
            text = text[:num_from] + current_number + text[num_to + 1:]
            return replace_text_numbers(text)
    return text.strip()


def clear_text_characters(text, exception_set=None, capitalize_words=False):
    """
    Clears text from all non-alphanumeric characters not in exception_set.
    Transforms non-latin characters to latin alternatives.
     """
    if exception_set is None:
        exception_set = {' ', '-', '.'}

    text = replace_text_numbers(text)

    text_clean = ''
    for char in text:
        if (char in exception_set) or (char.isalnum()):
            text_clean += char
        else:
            text_clean += ' '


    text = ''.join(c for c in unicodedata.normalize('NFKD', text_clean) if unicodedata.category(c) != 'Mn')

    manual_replacements = {
        'Œ': 'OE', 'œ': 'oe',
        'Æ': 'AE', 'æ': 'ae',
        'ß': 'ss',  # German sharp S
        'Ø': 'O', 'ø': 'o',  # Scandinavian O-slash
    }

    for original, replacement in manual_replacements.items():
        text = text.replace(original, replacement)

    text = re.sub(r' +', ' ', text).strip()
    if capitalize_words:
        text = ' '.join([word.capitalize() for word in text.split()])
        
    return text


def clear_corpus_characters(corpus, exclude_threshold: int = 100):
    """Clears corpus texts from all infrequent non-alphanumeric characters
    (frequency below threshold). Transforms all non-latin characters to
    latin alternatives, replaces numbers with words.
    """
    char_counts = defaultdict(int)
    for text in corpus:
        for character in text:
            char_counts[character] += 1
    exception_characters = {char for char in char_counts
                            if (char_counts[char] > exclude_threshold
                                and not char.isalnum())}
    for i in range(len(corpus)):
        corpus[i] = clear_text_characters(corpus[i], exception_characters)
    return corpus


def clean_bio(bio):
    """
    Removes structured metadata (e.g., "Sex: Female\nBirthday: ...") from the start of a bio.
    
    Args:
        bio (str): Original bio text.

    Returns:
        str: Cleaned bio without structured metadata.
    """
    pattern = r"^(?:[A-Za-z\s]+(([:-]\s)|\t)*[^\n]*\n)+"
    cleaned_bio = re.sub(pattern, "", bio, flags=re.MULTILINE).strip()
    return cleaned_bio