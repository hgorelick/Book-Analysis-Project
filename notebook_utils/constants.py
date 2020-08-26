__all__ = ['PROJ_ROOT', 'DATA_PATHS', 'GR_KEYS', 'GENRES', 'GENRE_COMBS', 'scaler', 'minmax', 'remove_punct', 'roget_thesaurus', 'get_dir',
           'GENRE_MAP', 'NEW_GENRES', 'NO_HORROR', 'NUM_GENRES', 'EMOTIONS', 'ALL_MODELS', 'MODEL_NAMES', 'NON_NGRAM',
           'WORD_CHOICE', 'CAPITALS', 'PRODUCTIONS']


import os
import pickle
import string
from itertools import combinations
from pathlib import Path

from sklearn import preprocessing

from roget.roget_thesaurus import RogetThesaurus


def get_dir(path: Path, root_file: str):
    for root, dirs, files in os.walk(str(path)):
        if root_file in files:
            return path
        break
    path = path.parent
    return get_dir(path, root_file)


PROJ_ROOT = get_dir(Path().cwd(), "README.md")

DATA_PATHS = {"all": PROJ_ROOT.joinpath("data", "all book data"),
              "book lengths": PROJ_ROOT.joinpath("data", "book lengths"),
              "processed list": PROJ_ROOT.joinpath("data", "all book data", "processed_list"),
              "text path": get_dir(PROJ_ROOT, "zipfileLinks.txt").joinpath("processed text files")}


def get_gr_api_keys(path: Path):
    """
    Gets your Goodreads api keys from the given file path.
    Must be in the following format:
        key: [key]
        secret: [secret]
    """
    lines = open(str(path), "r+").readlines()
    keys = {"public": lines[0].split(": ")[1], "secret": lines[1].split(": ")[1]}
    return keys


GR_KEYS = get_gr_api_keys(PROJ_ROOT)

# BOOK_NUMBERS = pickle.load(open(str(PROJ_ROOT.joinpath("data", "BOOK_NUMBERS")), "rb+"))

GENRES = ["Adventure_Stories", "Fiction", "Historical_Fiction",
          "Love_Stories", "Mystery", "Poetry", "Science_Fiction", "Short_Stories"]

scaler = preprocessing.MinMaxScaler()
minmax = preprocessing.minmax_scale
# rscaler = preprocessing.RobustScaler()

remove_punct = str.maketrans("", "", string.punctuation)

roget_thesaurus = RogetThesaurus(PROJ_ROOT.joinpath("roget", "roget_thesaurus.csv"))

GENRE_MAP = {"Adventure": "Adventure", "Adventure Stories": "Adventure",
             "Autobiographies": "(Auto)Biography", "Autobiography": "(Auto)Biography",
             "Biographies": "(Auto)Biography", "Biography": "(Auto)Biography",
             "Children's Fiction": "Children",
             "Children's Literature": "Children",
             "Children's Instructional Books": "Children",
             "Children's Myths, Fairy Tales, etc.": "Children",
             "Children's Picture Books": "Children",
             "Crime Fiction": "Detective", "Detective Fiction": "Detective",
             "Drama": "Drama",
             "Fantasy": "Fantasy",
             "Fiction": "Fiction", "General Fiction": "Fiction",
             "Historical Fiction": "Historical Fiction",
             "Horror": "Horror",
             "Humor": "Humor",
             "Juvenile Fiction": "Children", "Juvenile Literature": "Children",
             "Love": "Romance Fiction",
             "Mystery Fiction": "Detective",
             "Plays": "Drama",
             "Poetry": "Poetry",
             "Romance Fiction": "Romance Fiction",
             "Satire": "Humor",
             "Science Fiction": "Science Fiction",
             "Short Stories": "Short Stories",
             "Western": "Historical Fiction", "Western Fiction": "Historical Fiction"}

NEW_GENRES = list(set(GENRE_MAP.values()))
NEW_GENRES.remove("(Auto)Biography")
NEW_GENRES.sort()

NO_HORROR = [genre for genre in NEW_GENRES if genre != "Horror"]

NUM_GENRES = 13

GENRE_COMBS = [(c1, c2) for c1, c2 in combinations(NEW_GENRES, 2)]

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

ALL_MODELS = ["bigram", "clausal", "lex", "lexg", "liwc", "nonlex", "nonlexg", "nrc", "phrasal", "pos", "roget", "wordnet", "unigram"]
MODEL_NAMES = ["bigram", "clausal", "lex", "lexg", "liwc", "nonlex", "nonlexg", "nrc", "phrasal", "pos", "roget", "wordnet", "unigram"]
NON_NGRAM = ["clausal", "lex", "lexg", "liwc", "nonlex", "nonlexg", "nrc", "phrasal", "pos", "roget", "wordnet"]
WORD_CHOICE = ["clausal", "liwc", "nrc", "phrasal", "pos", "roget", "wordnet"]
PRODUCTIONS = ["lex", "lexg", "nonlex", "nonlexg"]
CAPITALS = dict(zip(MODEL_NAMES, ["Bigram", "Clausal", "Lex", "LexG", "LIWC", "Nonlex", "NonlexG", "NRC", "Phrasal", "POS", "Roget", "WordNet", "Unigram"]))
