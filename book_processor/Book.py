import os
import pickle
import re
import string
from zipfile import ZipFile

import nltk
from collections import Counter
from functools import reduce
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import pandas as pd
from selenium.webdriver.chrome.webdriver import WebDriver

from gutenberg_utils import gutenberg_scraper as g


class Book:
    __name__ = "Book"

    def __init__(self, path: Path, success: bool, genre: str, browser: WebDriver):
        self.book_number = path.stem
        try:
            self.first_1k_sentences = process_text(open(str(path), "r+").readlines())[:1000]
        except UnicodeDecodeError:
            self.first_1k_sentences = process_text(open(str(path), "r+", encoding="utf-8").readlines())[:1000]
        self.full_text = g.get_full_text(browser, self)
        self.num_words = len(self.full_text.split())
        self.num_unique_words = len(list(self.unigrams.keys()))
        self.success: str = "SUCCESSFUL" if success else "FAILURE"
        self.genre: str = genre
        self.full_pos_counts: Dict = self.full_text_pos_tag_counts()
        self.first_1k_pos_counts: Dict = self.first_1k_pos_tag_counts()

    def first_1k_sentences_pos_tags(self):
        # try:
        return [pair for pair in nltk.pos_tag(''.join(self.first_1k_sentences)).split()]
        # except:
        #     fix_unclosed_quotes(self.first_1k_sentences)
        #     return self.first_1k_sentences_pos_tags()

    def full_text_pos_tags(self):
        return [pair for pair in nltk.pos_tag(self.full_text.split())]

    def first_1k_pos_tag_counts(self, limit: Optional[int] = None) -> Dict:
        counts = Counter([tagged[1] for tagged in self.first_1k_sentences_pos_tags() if tagged[1][0] not in string.punctuation])
        return {pair[0]: pair[1] for pair in counts.most_common(limit)}

    def full_text_pos_tag_counts(self, limit: Optional[int] = None) -> Dict:
        counts = Counter([tagged[1] for tagged in self.full_text_pos_tags() if tagged[1][0] not in string.punctuation])
        return {pair[0]: pair[1] for pair in counts.most_common(limit)}

    @property
    def unigrams(self):
        return Counter([word for word in self.full_text.split()])

    def __repr__(self):
        return f"Book # {self.book_number}: {self.success}"

    def __str__(self):
        return f"Book # {self.book_number}: {self.success}"


# region UTILS

def process_text(text, full_text=False):
    text = ' '.join(text)
    text = g.remove_new_lines(text)

    if not full_text:
        lines = split_text(text)

        fix_person_names(lines, text, ["mr. s. ", "harlan p. "])
        fix_unclosed_quotes(lines)

        fixed = []
        for line in lines:
            if line != (". " or "." or " .") and not re.match("chapter [ivx].*$", line, re.IGNORECASE):
                fixed.append(line)

        return fixed

    return text


def fix_unclosed_quotes(lines):
    for (i, line) in enumerate(lines):
        unclosed_quote_idx = i
        if len(re.findall("(\"[^\"]*)(?!\")", line)) == 1 or len(re.findall("(\"[^\"]*)(?!\")", line)) > 2:
            unclosed_quote = lines.pop(unclosed_quote_idx)
            if unclosed_quote_idx == len(lines):
                lines[unclosed_quote_idx - 1] = unclosed_quote + lines[unclosed_quote_idx - 1]
            else:
                lines[unclosed_quote_idx] = unclosed_quote + lines[unclosed_quote_idx]
                i -= 1
                line = lines[i]


def fix_person_names(lines, text, names):
    for name in names:
        if name in text:
            initial_name_idx = [i for i, line in enumerate(lines) if name in line][0]
            initial_name_line = lines.pop(initial_name_idx)
            lines[initial_name_idx] = initial_name_line + lines[initial_name_idx]


def split_text(text):
    lines = [line for line in
             re.split("((?<=[.!?]\\s)|(?<=[.!?]\\\"\\s))(?<!mrs\\..)(?<!mr\\..)(?<!ms\\..)(?<!no\\..)(?<! [a-zA-z]\\..)", text)
             if line is not None and line != ""]
    return lines


def pos_tag_counts_by_genre(all_books_by_genre: Dict, all=True, book_idx=0, common=False, limit: Optional[int] = None) -> Tuple[Dict, List]:
    """
    """

    pos_by_genre_success_distr = {genre: {} for genre, books in all_books_by_genre.items()}
    tags: List[Dict] = []

    for genre, books in all_books_by_genre.items():
        genre_total_pos = 0.0
        if all:
            print("##### Tagging all {} #####".format(genre))
            for i in range(len(books)):
                all_books_by_genre[genre][i].pos_counts = get_pos_counts(all_books_by_genre[genre][i], genre, limit)

                for pos, count in all_books_by_genre[genre][i].pos_counts.items():
                    genre_total_pos += float(count)
                    success = all_books_by_genre[genre][i].success

                    if pos in list(pos_by_genre_success_distr[genre].keys()):
                        if success in list(pos_by_genre_success_distr[genre][pos].keys()):
                            pos_by_genre_success_distr[genre][pos][success] += float(count)
                        else:
                            pos_by_genre_success_distr[genre][pos].update({success: float(count)})
                    else:
                        pos_by_genre_success_distr[genre].update({pos: {success: float(count)}})
                tags.append(all_books_by_genre[genre][i].pos_counts)

            for pos, counts in pos_by_genre_success_distr[genre].items():
                try:
                    successful = float(pos_by_genre_success_distr[genre][pos]["SUCCESSFUL"])
                except KeyError:
                    pos_by_genre_success_distr[genre][pos].update({"SUCCESSFUL": 0})
                    successful = 0

                try:
                    failure = float(pos_by_genre_success_distr[genre][pos]["FAILURE"])
                except KeyError:
                    pos_by_genre_success_distr[genre][pos].update({"FAILURE": 0})
                    failure = 0

                pos_by_genre_success_distr[genre][pos] = (successful / genre_total_pos) - (failure / genre_total_pos)

        else:
            pos_counts = get_pos_counts(books[book_idx], genre, limit)
            pos_by_genre_success_distr.update({genre: pos_counts})
            tags.append(pos_counts)

        # pos_total += float(genre_total_pos)

    genres = list(all_books_by_genre.keys())

    pos_by_genre_success_distr_invert = get_all_pos_counts_per_genre(genres, pos_by_genre_success_distr, tags)
    return pos_by_genre_success_distr_invert, genres


def get_pos_counts(book: Book, genre, limit: Optional[int] = None) -> Dict:
    print("Tagging {} Book # {}...".format(genre, book.book_number))
    tag_counts = book.first_1k_pos_tag_counts(limit)
    print("{} Book # {}: {}\n".format(genre, book.book_number, tag_counts))
    return tag_counts


def get_common_pos_counts_per_genre(genres, pos_by_genre, tags):
    common_pos = reduce(lambda x, y: x.keys() & y.keys(), tags)
    common_pos_counts_per_genre = {k: {} for k in common_pos}
    for genre in genres:
        for pos_tags, count in pos_by_genre[genre].items():
            if pos_tags in common_pos:
                common_pos_counts_per_genre[pos_tags].update({genre: count})
    return common_pos_counts_per_genre


def get_all_pos_counts_per_genre(genres, pos_by_genre, tags):
    all_pos = reduce(lambda x, y: x.union(list(y.keys())), tags, set())
    pos_by_genre_invert = {pos: {genre: 0 for genre in genres} for pos in all_pos}
    for genre in genres:
        for pos_tag, count in pos_by_genre[genre].items():
            pos_by_genre_invert[pos_tag][genre] = count
    return pos_by_genre_invert


def get_pos_tags_per_genre_from_books(all_books):
    tags = [book.pos_counts for genre, books in all_books.items() for book in books]
    genres = list(all_books.keys())

    pos_by_genre = {genre: {} for genre, books in all_books.items()}
    for genre, books in all_books.items():
        for book in books:
            for pos_tags, count in book.pos_counts.items():
                if pos_tags in list(pos_by_genre[genre].keys()):
                    pos_by_genre[genre][pos_tags] += count
                else:
                    pos_by_genre[genre].update({pos_tags: count})

    return get_all_pos_counts_per_genre(genres, pos_by_genre, tags)

# endregion


# region Helpers
def get_proj_root(cwd: Path):
    for root, dirs, files in os.walk(str(cwd)):
        if "README.md" in files:
            return cwd
        break
    cwd = cwd.parent
    return get_proj_root(cwd)


PROJ_ROOT = get_proj_root(Path().cwd())


def load_books_from_genre(genre: str):
    if genre.lower().startswith("a"):
        return pickle.load(open("../data/books_by_genre.zip/all_Adventure_Stories_books.txt", "rb+"))
    if genre.lower().startswith("f"):
        return pickle.load(open("../data/books_by_genre.zip/all_Fiction_books.txt", "rb+"))
    if genre.lower().startswith("h"):
        return pickle.load(open("../data/books_by_genre.zip/all_Historical_Fiction_books.txt", "rb+"))
    if genre.lower().startswith("l"):
        return pickle.load(open("../data/books_by_genre.zip/all_Love_Stories_books.txt", "rb+"))
    if genre.lower().startswith("m"):
        return pickle.load(open("../data/books_by_genre.zip/all_Mystery_books.txt", "rb+"))
    if genre.lower().startswith("p"):
        return pickle.load(open("../data/books_by_genre.zip/all_Poetry_books.txt", "rb+"))
    if genre.lower().startswith("sc"):
        return pickle.load(open("../data/books_by_genre.zip/all_Science_Fiction_books.txt", "rb+"))
    if genre.lower().startswith("sh"):
        return pickle.load(open("../data/books_by_genre.zip/all_Short_Stories_books.txt", "rb+"))


def get_all_book_paths(root_dir):
    paths = {}
    for root, dirs, files in os.walk(root_dir):
        if "data" in dirs:
            return get_all_book_paths(str(PROJ_ROOT.joinpath("data")))
        if "novels" in dirs:
            return get_all_book_paths(str(PROJ_ROOT.joinpath(root, "novels")))
        for file in files:
            if "novel_meta.txt" not in files and file.endswith(".txt"):
                path = Path().joinpath(root, file)
                if path.parts[-4] not in paths.keys():
                    paths.update({path.parts[-4]: [path]})
                else:
                    paths[path.parts[-4]].append(path)
    return paths


def get_all_books_from_zip(path_to_zip: Path = Path().joinpath(PROJ_ROOT, "data"), file_name: str = "books_by_genre.zip",
                           as_list=True) -> Union[List, Dict]:

    print("\nLoading books from {}...".format(str(path_to_zip.joinpath(file_name))))
    z = ZipFile(str(path_to_zip.joinpath(file_name)))
    namelist = z.namelist()
    if as_list:
        all_books = []
        for path in namelist:
            books = pickle.load(z.open(path))
            all_books += books
    else:
        all_books = {}
        for path in namelist:
            all_books.update({re.search("(?<=all_).*(?=_books)", path)[0]: pickle.load(z.open(path))})
    print("Books loaded!\n")
    return all_books


def dump_books(all_books: List['Book'], path: str):
    if os.path.exists(path):
        i = 1
        while os.path.exists("{}_{}.txt".format(path, i)):
            i += 1
        file = "{}_{}".format(path, i)
        print("\n######## Dumping books as pickle object to {} ########".format(file))
        with open("{}".format(file), "wb+") as f:
            try:
                pickle.dump(all_books, f)
            except MemoryError:
                print("There was a MemoryError when dumping {}".format(file))

    else:
        print("\n######## Dumping books as pickle object to {} ########".format(path))
        with open("{}".format(path), "wb+") as f:
            try:
                pickle.dump(all_books, f)
            except MemoryError:
                print("There was a MemoryError when dumping {}".format(path))


def dump_books_by_genre(all_books: Union[Dict, List], genre=None, file_name_mod: str = ".txt"):
    if genre is not None:
        cwd = Path().cwd()
        path = Path().joinpath(cwd, "data", "books_by_genre", f"all_{genre}_books{file_name_mod}")
        dump_books([book for book in all_books if book.genre == genre], path)
    else:
        for genre, books in all_books.items():
            dump_books(books, str(PROJ_ROOT.joinpath("data", "books_by_genre", f"all_{genre}_books{file_name_mod}")))


def load_books_by_genre(books_by_genre_path: str, as_list=True):
    if as_list:
        all_books = []
        for path in get_all_books_from_zip(books_by_genre_path):
            books = pickle.load(open(path, "rb+"))
            all_books += books
    else:
        all_books = {}
        for path in get_all_books_from_zip(books_by_genre_path):
            key = re.search("(?<=all_).*(?=_books)", path.stem)[0]
            val = pickle.load(open(path, "rb+"))
            all_books.update({key: val})
    return all_books


def books_to_csv(books: List['Book'], fields: List[str]):
    print("Writing books to csv")
    df = pd.DataFrame([{fn: getattr(b, fn) for fn in fields} for b in books])
    cwd = Path().cwd()
    path = Path().joinpath(cwd, "data", "all_books.csv")
    all_books_csv = open(path, 'w+', newline='')
    df.to_csv(all_books_csv, index=False)
    print("Complete!")


def dump_pos_tags(pos_tags: Dict, path: Path = Path().joinpath(PROJ_ROOT, "data", "pos_data"),
                  file_name: str = "pos_tag_counts_per_genre.txt", file_name_mod: Optional[str] = None):
    """
    """
    if file_name_mod is not None:
        file_name = file_name_mod + file_name
    print("\n######## Dumping POS tag counts by genre as pickle object to {} ########".format(str(path.joinpath(file_name))))
    with open(str(path.joinpath(file_name)), "wb+") as f:
        try:
            pickle.dump(pos_tags, f)
        except MemoryError:
            print("There was a MemoryError when dumping {}".format(str(path.joinpath(file_name))))


def load_pos_tags(path: Path = Path().joinpath(PROJ_ROOT, "data", "pos_data"),
                  file_name: str = "pos_tag_counts_per_genre.txt"):
    pos_tags = pickle.load(open("{}".format(str(path.joinpath(file_name))), "rb+"))
    return pos_tags
# endregion
