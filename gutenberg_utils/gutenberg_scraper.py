import re

from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.webdriver import WebDriver

GUTENBERG_URL = "https://www.gutenberg.org/files/"


def get_full_text(browser: WebDriver, book):
    re_first_sentence = get_re_pattern(book.first_1k_sentences[0])
    link = get_txt_link(book.book_number, browser)

    text = get_text(book.book_number, browser, link)
    try:
        full_text_cleaned = clean_text(text)
        first_word_idx: int = re.search(re_first_sentence, full_text_cleaned).start()
    except AttributeError as e:
        return ["-1"]

    text = remove_gutenberg_notes(full_text_cleaned[first_word_idx:])
    return "".join(text)


def remove_gutenberg_notes(text: str):
    lines = [line for line in
             re.split("((?<=[.!?]\\s)|(?<=[.!?]\\\"\\s))(?<!mrs\\..)(?<!mr\\..)(?<!ms\\..)(?<!no\\..)", text)
             if line is not None and line != ""]
    if len(lines) == 0:
        print("")
    for j, sentence in enumerate(lines):
        if "End of the Project Gutenberg" in sentence:
            lines = lines[:j]
            if len(lines) == 0:
                print("")
            return lines
    if len(lines) == 0:
        print("")
    return lines


def get_text(book_number, browser, link):
    loaded = False
    text = ""
    while not loaded:
        try:
            browser.get("{}{}/{}".format(GUTENBERG_URL, book_number, link.attrs["href"]))
            soup = BeautifulSoup(browser.page_source, "lxml")
            text = str(soup.find("body").text)
            text = remove_new_lines(text)
            loaded = True
        except TimeoutException:
            continue
    return text


def remove_new_lines(text):
    lines = [line for line in text.split("\n")]
    cleaned_lines = []
    for line in lines:
        if len(re.findall("chapter ([ivx]+)", "", line)) == 0:
            cleaned_line = ""
            while "\n" in line or len(re.findall("  +", line)) > 0:
                cleaned_line = " ".join(line.split("\n"))
                cleaned_line = re.sub("  +", " ", cleaned_line)
            cleaned_lines.append(cleaned_line)
    return "".join(cleaned_lines)


def get_txt_link(book_number, browser: WebDriver):
    loaded = False
    link = ""
    while not loaded:
        try:
            browser.get("{}{}".format(GUTENBERG_URL, book_number))
            soup = BeautifulSoup(browser.page_source, "lxml")
            link = [link for link in soup.find_all("a") if ".txt" in link.attrs["href"]][0]
            loaded = True
        except (TimeoutException, IndexError) as e:
            continue
    return link
    # try:
    # except IndexError as e:
    #     with open(str(Book.PROJ_ROOT.joinpath("errors.txt")), "w+") as f:
    #         f.write(f"Error writing book # {book_number}.\nError:\n{e}")
    # return link


def get_re_pattern(first_sentence: str):
    first_sentence = clean_text(first_sentence, True)
    return re.compile(first_sentence.strip(), re.IGNORECASE)


def clean_text(text: str, full: bool = False):
    cleaned_text = re.sub("’", "'", re.escape(text) if full else text)
    cleaned_text = re.sub("â€™", "'", cleaned_text)
    cleaned_text = re.sub("â€˜", "'", cleaned_text)
    cleaned_text = re.sub("‘", "'", cleaned_text)
    cleaned_text = re.sub("“", '"', cleaned_text)
    cleaned_text = re.sub("”", '"', cleaned_text)
    cleaned_text = re.sub("â€\"", "--", cleaned_text)
    cleaned_text = re.sub("Ã©", "e", cleaned_text)
    cleaned_text = re.sub("é", "e", cleaned_text)
    cleaned_text = re.sub("è", "e", cleaned_text)
    cleaned_text = re.sub("ñ", "n", cleaned_text)
    cleaned_text = re.sub("â", "a", cleaned_text)
    cleaned_text = re.sub("An°", "Anno", cleaned_text)
    cleaned_text = re.sub("Ï", "I", cleaned_text)
    cleaned_text = re.sub("_", "", cleaned_text)
    cleaned_text = re.sub("—", "--", cleaned_text)
    if full:
        for replace_str in set(re.findall("\\\\\\s", cleaned_text) + ["\\-"]):
            cleaned_text = re.sub(re.compile(re.escape(replace_str)), replace_str[-1], cleaned_text)
        return remove_new_lines(cleaned_text)
    return cleaned_text
