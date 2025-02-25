import sys
import argparse
import io
import json
import logging
import os
import re
import time

import imagehash
import pandas as pd
import requests
from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from tqdm import tqdm


def get_default_chrome_options():
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    return options


def beautify_bio(bio):
    if not bio:
        return ""
    
    bio = bio.strip()
    bio = re.sub(r"\(Source: .+?\)$", "", bio)
    bio = "\n".join(bio.splitlines())
    bio = re.sub(r'\n+', '\n', bio).strip()

    if bio == "No biography written.":
        bio = ""
    return bio


def get_image(image_link, image_path, config):
    """ Download an iamge and save it under a unique hash-name """
    image_content = requests.get(image_link, headers=config['headers']).content
    image_bytes = io.BytesIO(image_content)
    image = Image.open(image_bytes)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image_hash = str(imagehash.average_hash(image, hash_size=12))
    image_name = f"{image_path}/{image_hash}.jpg"
    image.save(image_name)
    return image_hash


def extract_links_from_page(source):
    """ Extracts all character links from MAL html source """
    soup = BeautifulSoup(source, "html.parser")
    table = soup.find('table')
    row_hrefs = [row['href'] for row in table.findAll('a', href=True)]
    char_links = []
    for href in row_hrefs:
        if href.startswith('https://myanimelist.net/character/') and href not in char_links:
            char_links.append(href)
    return char_links


def download_links(link_path):
    """ Downloads all character links from MyAnimeList using Selenium """
    if os.path.exists(link_path):
        with open(link_path, 'r') as file:
            character_links = [line for line in file.readlines() if line]
    else:
        character_links = []
    show = 0
    if character_links:
        link, show = character_links[-1].split('\t')
        show = int(show) + 50
        starting_letter = link.split('/')[-1].split('_')[-1][0]

    else:
        starting_letter = "A"

    starting_link = f"https://myanimelist.net/character.php?letter={starting_letter}&show={show}"

    letter_links = [f"https://myanimelist.net/character.php?letter={letter}"
                    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    if letter > starting_letter]  # resume from the latest letter
    
    links_to_parse = [starting_link] + letter_links

    print('Resuming from letter', starting_letter, 'showing', show)

    options = get_default_chrome_options()
    options.page_load_strategy = 'eager'

    driver = webdriver.Chrome(options=options)  # optional argument, if not specified will search path.

    with open(link_path, 'a') as file:
        for link in links_to_parse:
            driver.get(link)

            click_next = int(show / 50) + 2

            while True:
                try:
                    page_character_links = extract_links_from_page(driver.page_source)
                    show += 50
                    for character_link in page_character_links:
                        file.write(f"{character_link}\t{show}\n")
                    pages = driver.find_element(By.CLASS_NAME, "normal_header")
                    next_page = pages.find_element(By.LINK_TEXT, str(click_next))
                    next_page.click()
                    time.sleep(2)
                    click_next += 1
                except NoSuchElementException:
                    show = 0
                    break

    driver.quit()


def download_characters(data_path, link_path, img_dir, config):
    """ Downloads all character data and images using JikanAPI for MAL """

    session_count = 0
    start = 0

    with open(link_path, 'r') as file:
        links = file.readlines()

    if os.path.exists(data_path):
        mal_characters = pd.read_csv(data_path).fillna('')
    else:
        mal_characters = pd.DataFrame(columns=['eng_name', 'kanji_name', 'bio', 'mal_link', 'img_link', 'img_index'])
    start = len(mal_characters)
    if session_count == 0:
        print(f'\nWelcome back!\nYou have already parsed {start} / {len(links)} characters.'
                f'\n{len(links) - start} more to go!\nGood luck!\n')

    current_time = time.time()
    
    for index, link in tqdm(enumerate(links[start:], start=start), total=len(links[start:])):
        while True:  # basic retry iteration after an exception (in this case, URLError)
            try:
                character_id = link.split('/')[-2]
                character = requests.get(f'https://api.jikan.moe/v4/characters/{character_id}').json()['data']
                mal_characters.at[index, 'eng_name'] = character['name']
                mal_characters.at[index, 'kanji_name'] = character['name_kanji']
                mal_characters.at[index, 'bio'] = beautify_bio(character['about'])
                mal_characters.at[index, 'mal_link'] = character['url']

                image_url = character['images']['jpg']['image_url']
                if image_url.endswith('.jpg'):
                    mal_characters.at[index, 'img_link'] = image_url
                    mal_characters.at[index, 'img_index'] = get_image(image_url, img_dir, config)
                else:
                    mal_characters.at[index, 'img_index'] = "-1"

                session_count += 1

            except KeyboardInterrupt:  # save when interrupted
                mal_characters.to_csv(data_path, index_label=False)
                print('Interrupted by user!')
                sys.exit()

            except Exception as E:  # any other exception, log it and continue
                logging.error(f"{link} failed to fetch with {str(E)}")
                mal_characters.to_csv(data_path, index_label=False)
                print(f"{link} failed to fetch with {str(E)}")
                session_count += 1

            finally:  # save every 60 requests + wait out time to avoid rate limit
                if session_count % 60 == 0 and session_count != 0:
                    mal_characters.to_csv(data_path, index_label=False)
                    elapsed_time = time.time() - current_time
                    if elapsed_time < 60:
                        wait_for = 60 - elapsed_time
                        print('Waiting for', wait_for, 'seconds...')
                        time.sleep(wait_for)  # because we can have maximum 60 requests per minute
                    current_time = time.time()

            break

    mal_characters.to_csv(data_path, index_label=False)  # save the last batch


def main(arguments):

    logging.basicConfig(filename=arguments.log_path,
                        format='%(asctime)s %(message)s',
                        level=logging.ERROR)

    with open(arguments.config_path) as extract_config:
        config = json.load(extract_config)

    if not os.path.isdir(arguments.img_dir):
        os.mkdir(arguments.img_dir)

    if arguments.get_links:
        print('Extracting links first!')
        download_links(arguments.link_path)
        print('Extracting links finished! Launch the script without --get_links next.')
    else:
        print('Extracting character data!')
        download_characters(arguments.data_path, arguments.link_path, arguments.img_dir, config)
        print('All done!')


def parse_arguments():
    parser = argparse.ArgumentParser(description='This script downloads MyAnimeList data '
                                                 'from character links using v4 of Jikan API.')
    parser.add_argument('--get_links', action='store_true',
                        help='download character links first')
    parser.add_argument('--link_path', default='./links.txt', metavar='LINK_PATH',)
    parser.add_argument('--data_path', default='./anime_characters.csv', metavar='DATA_PATH',
                        help='set csv file for saving character data')
    parser.add_argument('--img_dir', default='./images', metavar='IMG_PATH',
                        help='set directory for saving character images')

    parser.add_argument('--config_path', default='./config/extract.json', metavar='CONFIG_PATH',
                        help='set config for requests library file path')
    parser.add_argument('--log_path', default='./logs/get_pages.log', metavar='LOG_PATH',
                        help='set logging file path')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
