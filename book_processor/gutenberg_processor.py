import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
import re
from typing import List, TextIO, Optional, Tuple, Dict, Union

from bs4 import BeautifulSoup as bs, Tag, NavigableString, Comment
import nltk
nltk.data.path.append("/mnt/Public/mamin17/hgorelick/nltk_data")

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from tqdm import tqdm

from notebook_utils.constants import get_dir, GENRE_MAP, NEW_GENRES, PROJ_ROOT

logging.basicConfig(filename="gutenberg_processor.log", filemode="w", format="%(asctime)s::%(levelname)s - %(message)s", level=logging.INFO)

sys.path.append("pydevd-pycharm.egg")

tqdm.pandas()

# NEW_GENRES_KEYS = ["Adventure", "Adventure Stories",
#               "Autobiographies", "Autobiography", "Biographies", "Biography",
#               "Children's Fiction", "Children's Literature", "Children's Myths, Fairy Tales, etc.", "Crime Fiction",
#               "Detective Fiction", "Drama", "Fantasy", "Fiction", "General Fiction", "Historical Fiction", "Horror",
#               "Humor", "Juvenile Fiction", "Juvenile Literature", "Love", "Mystery Fiction", "Plays", "Poetry",
#               "Romance Fiction", "Science Fiction", "Short Stories", "Western", "Western Fiction"]

COMPOUND_GENRES = ["Children's Fiction", "Crime Fiction", "Detective Fiction", "General Fiction", "Historical Fiction",
                   "Juvenile Fiction", "Mystery Fiction", "Romance Fiction", "Science Fiction", "Western Fiction"]

NEW_GENRES_KEYS = list(GENRE_MAP.keys())

NEW_GENRES_LOWER = {genre.lower(): genre for genre in NEW_GENRES_KEYS}


def try_soup(html: TextIO, book_number: str):
    try:
        soup = bs(html, "lxml")
    except UnicodeDecodeError:
        logging.error(f"B# {book_number} -- bs4 encoding error, trying with ISO-8859-1", exc_info=True)
        try:
            soup = bs(html, parser="lxml", features="lxml", from_encoding="latin-1")
        except UnicodeDecodeError:
            logging.error(f"B# {book_number} -- bs4 encoding error", exc_info=True)
            return None
    return soup


def open_file(path: Path):
    text = open(str(path), "r+", encoding="utf-8")
    try:
        _ = text.readlines()
        text = open(str(path), "r+", encoding="utf-8")
    except UnicodeDecodeError:
        text = open(str(path), "r+", encoding="ISO-8859-1")
    return text


def process_html(path: Path, book_number: str, punch: bool = False):
    html = open_file(path)
    soup = try_soup(html, book_number)

    try:
        if len(soup.contents) == 0:
            html = open(str(path), "r+", encoding="ISO-8859-1")
            soup = try_soup(html, book_number)
    except AttributeError:
        return [""]

    if punch:
        text = get_punch_text(soup, book_number)

    elif len(soup.find_all(name="div", class_="poem")) > 0:
        text = get_poem_text(soup, book_number)

    elif book_number != "1154" and len(soup.find_all(name="div", class_="chapter")) > 0:
        text = get_chapter_text(soup, book_number)

    else:
        text = get_generic_text(soup, book_number)

    if len(text) < 4 or all(s == "" for s in text):
        text = get_table_text(soup, book_number)

    if len(text) < 4 or all(s == "" for s in text):
        text = get_div_text(soup, book_number)

    if len(text) < 4 or all(s == "" for s in text):
        return [""]

    if isinstance(text, list):
        joined = "\n".join(text)
    else:
        joined = text

    return process_text(joined, book_number)


def get_poem_text(soup: bs, book_number: str):
    poem_text = []
    for poem in soup.find_all("div", class_="poem"):
        if len(poem.find_all("a")) > 0:
            continue
        poem_text.append(poem.get_text().strip())
    if len(poem_text) < 4:
        logging.error(f"B# {book_number} -- Poem text is less than 4 lines long")
    return poem_text


def get_chapter_text(soup: bs, book_number: str):
    chapter_text = []
    for chapter in soup.find_all("div", class_="chapter"):
        chapter_text.append(get_generic_text(chapter, book_number))
    if len(chapter_text) < 4:
        logging.error(f"B# {book_number} -- Chapter text is less than 4 lines long")
    return chapter_text


def get_div_text(soup: bs, book_number: str):
    div_text = get_generic_text(soup, book_number, "div", regex=False)
    if len(div_text) < 4:
        logging.error(f"B# {book_number} -- Div text text is less than 4 lines long")
    return div_text


def get_generic_text(soup: bs, book_number: str, tag: str = "^h[2-6]$", regex: bool = True):
    generic_text = []
    for header in soup.find_all(re.compile(tag) if regex else tag):
        next_node = header
        while True:
            next_node = next_node.nextSibling
            if next_node is None:
                break
            if isinstance(next_node, Tag):
                if "h" in next_node.name:
                    break
                elif next_node.name != "p":
                    continue
                elif next_node.name == "p":
                    if "class" in next_node:
                        if next_node["class"][0] == "folio" or next_node["class"][0] == "fnote":
                            continue
                    elif "class" in next_node.attrs.keys():
                        if next_node.attrs["class"][0] == "pfirst" or next_node.attrs["class"][0] == "pnext":
                            generic_text.append(next_node.get_text().strip())
                            continue
                if all(isinstance(c, NavigableString) or c.name == "br" for c in next_node.contents):
                    generic_text.append(next_node.get_text().strip())
                else:
                    generic_text.append("".join([text for text in next_node.contents if
                                                 isinstance(text, NavigableString) and not isinstance(text, Comment)]))
    if len(generic_text) < 4:
        logging.error(f"B# {book_number} -- Generic text is less than 4 lines long")
    return "\n".join(generic_text)


def get_punch_text(soup: bs, book_number: str):
    punch_text = []
    pts: List[bs] = soup.find_all("table")
    bordered = soup.find_all("table", attrs={"border": 1})

    try:
        idx = pts.index(bordered[1]) + 1
    except IndexError:
        return [""]

    punch_table = None

    for k in range(idx, len(pts)):
        if pts[k].has_attr("width") and pts[k].has_attr("align"):
            if pts[k].attrs["width"] == "800" and pts[k].attrs["align"] == "center":
                punch_table = pts[k]
                break

    if punch_table is None:
        return [""]

    for paragraph in punch_table.find_all("p"):
        next_node = paragraph
        while True:
            next_node = next_node.nextSibling
            if next_node is None:
                break
            if isinstance(next_node, Tag):
                if "h" in next_node.name:
                    break
                elif next_node.name != "p":
                    continue
                elif next_node.name == "p":
                    if "class" in next_node:
                        if next_node["class"][0] == "folio" or next_node["class"][0] == "fnote":
                            continue
                if all(isinstance(c, NavigableString) or c.name == "br" for c in next_node.contents):
                    punch_text.append(next_node.get_text().strip())
                else:
                    punch_text.append("".join([text for text in next_node.contents if
                                               isinstance(text, NavigableString) and not isinstance(text, Comment)]))
    if len(punch_text) < 4:
        logging.error(f"B# {book_number} -- Punch text is less than 4 lines long")
    return punch_text


def get_table_text(soup: bs, book_number: str):
    table_text = []
    for table in soup.find_all("tbody"):
        for row in table.find_all("tr"):
            for cell in row.find_all("td"):
                table_text.append("".join(get_generic_text(cell, book_number)))
    if len(table_text) < 4:
        logging.error(f"B# {book_number} -- Table text is less than 4 lines long")
    return table_text


def process_text(text: str, book_number: str):
    try:
        cleaned_text = clean_text(text)
        sentences = nltk.sent_tokenize(cleaned_text)
        gut_removed = remove_gutenberg_notes(sentences)
        return "\n".join([s for s in gut_removed if len(s) > 1])
    except AttributeError as e:
        logging.error(f"B# {book_number} -- Processing Error", exc_info=True)
        return None


def remove_new_lines(text: str):
    lines = text.split("\n")
    cleaned_lines = []

    for i in range(len(lines)):
        if not re.search("(\\*+\\s+)+", lines[i], re.IGNORECASE) and lines[i] != "" and "\xa0" not in lines[i]:
            cleaned_line = f"{lines[i].lstrip(' ')} "
            while "\n" in cleaned_line or len(re.findall("\\s\\s+", cleaned_line)) > 0:
                cleaned_line = " ".join(cleaned_line.split("\n"))
                cleaned_line = re.sub("\\s\\s+", " ", cleaned_line)
            cleaned_lines.append(cleaned_line)
    return re.sub("\\\\", "", "".join(cleaned_lines))


def clean_text(text: str):
    cleaned_text = re.sub("’", "'", re.escape(text))
    cleaned_text = re.sub("‘", "'", cleaned_text)
    cleaned_text = re.sub("â€™", "'", cleaned_text)
    cleaned_text = re.sub("â€˜", "'", cleaned_text)
    cleaned_text = re.sub("“", '"', cleaned_text)
    cleaned_text = re.sub("”", '"', cleaned_text)
    cleaned_text = re.sub("â€\"", "--", cleaned_text)
    cleaned_text = re.sub("Ã©", "e", cleaned_text)
    cleaned_text = re.sub("é", "e", cleaned_text)
    cleaned_text = re.sub("è", "e", cleaned_text)
    cleaned_text = re.sub("ñ", "n", cleaned_text)
    cleaned_text = re.sub("â\x80\x99", "'", cleaned_text)
    cleaned_text = re.sub("â", "a", cleaned_text)
    cleaned_text = re.sub("An°", "Anno", cleaned_text)
    cleaned_text = re.sub("Ï", "I", cleaned_text)
    cleaned_text = re.sub("_+", "", cleaned_text)
    cleaned_text = re.sub("—", "--", cleaned_text)
    cleaned_text = re.sub("Â", "", cleaned_text)
    cleaned_text = re.sub("a", "'", cleaned_text)
    cleaned_text = re.sub("\\*", "", cleaned_text)

    for replace_str in set(re.findall("\\\\\\s", cleaned_text) + ["\\-"]):
        cleaned_text = re.sub(re.compile(re.escape(replace_str)), replace_str[-1], cleaned_text)
    return remove_new_lines(cleaned_text)


def fix_unclosed_quotes(lines: List[str]):
    k = 0
    fixed = []
    while k in range(len(lines)):
        if len(re.findall("\"", lines[k])) % 2 == 1 or (len(re.findall("\"", lines[k])) % 2 == 0 and lines[k + 1][0].islower()):
            unclosed_quote = lines.pop(k)
            if k == len(lines):
                lines[k - 1] = f"{unclosed_quote} {lines[k - 1]}"
            else:
                lines[k] = f"{unclosed_quote} {lines[k]}"
                continue
        fixed.append(lines[k])
        k += 1
    return " ".join(fixed)


def fix_person_names(lines: List[str], text: str, names: List[str]):
    for name in names:
        if name in text:
            initial_name_idx = [i for i, line in enumerate(lines) if name in line][0]
            initial_name_line = lines.pop(initial_name_idx)
            lines[initial_name_idx] = initial_name_line + lines[initial_name_idx]


def remove_gutenberg_notes(lines: List[str]):
    start = 0
    for j, sentence in enumerate(lines):
        if "START OF THIS PROJECT GUTENBERG" in sentence:
            start = j + 1
        if "End of the Project Gutenberg" in sentence:
            lines = lines[start:j]
            return lines
    return lines[start:]


def load_rdf_df():
    if os.path.exists(str(PROJ_ROOT.joinpath("data", "rdf_data", "processed rdf data"))):
        rdf_data_df = pickle.load(open(str(PROJ_ROOT.joinpath("data", "rdf_data", "processed rdf data")), "rb+"))
        rdf_data_df.rename(columns={"Genre": "@Genre", "Downloads": "@Downloads"}, inplace=True)
        return rdf_data_df
    return None


def process_rdf_df(rdf_path: Optional[str] = None) -> Union[Tuple[pd.DataFrame, Dict], pd.DataFrame]:
    rdf_data_df = load_rdf_df()
    if rdf_data_df is not None:
        return rdf_data_df

    if rdf_path is None:
        rdf_path = PROJ_ROOT.joinpath("data", "rdf_data", "extracted_rdf_data.csv")

    rdf_df = pd.read_csv(open(str(rdf_path), "r+", encoding="utf-8"), index_col=False)

    filtered = rdf_df.copy().drop(columns=["Min Pub Year", "Max Pub Year"])
    filtered = filtered[filtered["Language"] == "en"]
    filtered.Genres = filtered.Genres.astype(str)
    filtered.Subjects = filtered.Subjects.astype(str)
    # filtered = filtered[filtered["Downloads"] >= 50]

    print("Processing RDF Data...")
    has_genre = filtered.progress_apply(process_genres, axis=1)
    processed_genres = has_genre.loc[has_genre["Genres"].notnull()]\
                                .explode("Genres")\
                                .reset_index(drop=True)\
                                .rename(columns={"Genres": "Genre"})

    by_genre = {genre: processed_genres[processed_genres["Genre"] == genre] for genre in NEW_GENRES}
    return processed_genres, by_genre


def process_genres(x: pd.Series):
    if x["Genres"] is not None:
        genres = [genre.lower() for genre in x["Genres"].split(";")]
        x["Genres"] = [NEW_GENRES_LOWER[genre] for genre in list(set(genres).intersection(set(NEW_GENRES_LOWER.keys())))]
    else:
        x["Genres"] = []

    subjects = [subj.split(" -- ") for subj in x["Subjects"].split(";")]
    subjects = [subj.lower() for subject in subjects for subj in subject]
    x["Genres"] += [NEW_GENRES_LOWER[genre] for genre in list(set(subjects).intersection(set(NEW_GENRES_LOWER.keys())))]

    subjects = [subj.split(", ") for subj in subjects]
    subjects = [subj.lower() for subject in subjects for subj in subject]
    x["Genres"] += [NEW_GENRES_LOWER[genre] for genre in list(set(subjects).intersection(set(NEW_GENRES_LOWER.keys())))]

    subjects = [subj.split() for subj in subjects]
    subjects = [subj.lower() for subject in subjects for subj in subject]
    x["Genres"] += [NEW_GENRES_LOWER[genre] for genre in list(set(subjects).intersection(set(NEW_GENRES_LOWER.keys())))]

    x["Genres"] = list(set(x["Genres"]))

    if len(x["Genres"]) == 0:
        x["Genres"] = None
        return x

    if len(set(x["Genres"]).intersection(set(COMPOUND_GENRES))) > 0 and "Fiction" in x["Genres"]:
        x["Genres"].remove("Fiction")

    x["Genres"] = list(set(GENRE_MAP[genre] for genre in x["Genres"]))
    return x


def make_processed_dir(path: Path, genre: str, loc: str):
    if not os.path.exists(str(path.joinpath(loc))):
        os.makedirs(str(path.joinpath(loc)))
    if not os.path.exists(str(path.joinpath(loc, genre))):
        os.makedirs(str(path.joinpath(loc, genre)))


def get_processed(path: Path):
    ps = []
    for r, ds, fs in os.walk(str(path)):
        for file in fs:
            if "bad books" not in file:
                try:
                    text = open(str(path.joinpath(file)), "r+").readlines()
                except OSError:
                    logging.error(f"OSError reading {path.joinpath(file)}")
                    continue
                if file.endswith(".txt") and len(text) > 5 and not all(len(s) == 1 for s in text[1].translate(text[1]).split()):
                    ps.append(file.split(".")[0])
    return list(set(ps))


def process_gutenberg_files(genre: str, rdf_data: pd.DataFrame, file_type: str):
    rdf_data_df = rdf_data[rdf_data["Genre"] == genre]
    book_numbers = rdf_data_df["Book #"].unique()
    book_numbers.sort()
    book_numbers = book_numbers.astype(str)

    # all_genres = rdf_data_df["Genre"].unique()
    # all_genres.sort()

    loc = "files" if file_type == "html" else "text files"
    proc_loc = "processed files" if file_type == "html" else "processed text files"

    files_path = get_dir(Path().cwd(), "zipfileLinks.txt")
    make_processed_dir(files_path, genre, proc_loc)

    # print(f"\nSkipping already processed {genre} books...")
    existing = get_processed(files_path.joinpath("processed files", genre)) + get_processed(files_path.joinpath("processed text files", genre))
    existing = list(set(existing))
    existing.sort()

    exclude = ["30282", "42713", "12472", "12905", "36020"]

    for root, dirs, files in os.walk(str(files_path.joinpath(loc))):
        files.sort()

        with tqdm(total=len(files), position=NEW_GENRES.index(genre)) as pbar:
            for i, book_file in enumerate(files):
                pbar.set_postfix_str(f"-- {genre}")
                b_num = book_file.split(".")[0].split("-h")[0]
                if b_num not in exclude and b_num not in existing and b_num in book_numbers:
                    if "chap" in b_num:
                        number = 12233
                    elif b_num == "Met_I-III" or b_num == "Met_IV-VII":
                        number = 21765
                    elif b_num == "Met_VIII-XI" or b_num == "Met_XII-XV":
                        number = 26073
                    elif "s37" in b_num:
                        number = 10020
                    else:
                        number = int(b_num)
                    title = rdf_data_df.loc[rdf_data_df["Book #"] == number, "Title"].values[0]
                    # pbar.set_postfix_str(f"-- [{b_num}, {title}")
                    fpath = files_path.joinpath(loc, book_file)
                    if file_type == "html":
                        processed = process_html(fpath, b_num, "Punchinello" in title)
                    else:
                        text = open_file(fpath).readlines()
                        processed = process_text("".join(text), b_num)
                    if " " not in processed or len(processed.split()) < 5:
                        pbar.update(1)
                        if genre != "Poetry" and genre != "Children":
                            logging.info(f"B# {b_num} -- bad processing")
                        continue
                    with open(str(files_path.joinpath(proc_loc, genre, f"{b_num}.txt")), "w+") as f:
                        f.writelines(processed)
                pbar.update(1)
