import logging
import os
import pickle
import re
import string
import sys
sys.path.append(os.getcwd())

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import nltk
nltk.data.path.append("/mnt/Public/mamin17/hgorelick/nltk_data")
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.grammar import Production, Nonterminal
from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree, Tree, _child_names
from requests.exceptions import HTTPError
from tqdm import tqdm

from notebook_utils.constants import PROJ_ROOT, EMOTIONS, ALL_MODELS, roget_thesaurus, remove_punct, NON_NGRAM
from loading_utils.data_loader import load_liwc, load_nrc, DataLoader

ps = nltk.PorterStemmer()
LIWC = load_liwc(ps)
NRC = load_nrc()

clausal = ["S", "SBAR", "SBARQ", "SINV", "SQ"]
phrasal = ["ADJP", "ADVP", "CONJP", "FRAG", "INTJ", "LST", "NAC", "NP", "NX", "PP", "PRN", "PRT", "QP", "RRC", "UCP", "VP",
           "WHADJP", "WHAVP", "WHNP", "WHPP", "X"]


def get_wn_pos(pos: str) -> Optional[Tuple]:
    if pos[0] == "J":
        return "a", "s"
    elif pos[0] == "V":
        return "v",
    elif pos[0] == "N":
        return "n",
    elif pos[0] == "R":
        return "r",
    return None


def get_wn_words(word: str):
    return [synset.name().split(".")[0] for synset in wn.synsets(word)]


def get_liwc_category(w: str):
    cats = []
    for cat, stems in LIWC.items():
        if w in stems:
            cats.append(cat)
    return cats


def get_counts(data_list: List[Dict]):
    return {name: sum([Counter(data_list[i][name]) for i in range(len(data_list))], Counter())
            for name in ALL_MODELS if "gram" not in name}


def prep_for_df(data_dict: Dict, counts: Dict, model_name: str):
    data_dict.update({k: v for k, v in counts[model_name].items() if k != "''" and k != "``" and k not in string.punctuation})


def make_data_dirs(path: Path):
    for dir_name in ["whole", "first 1k"]:
        if not os.path.exists(str(path.joinpath(dir_name))):
            os.makedirs(str(path.joinpath(dir_name)))


def write_book_data(path: Path, book_number: str, book_data: List, counts: List):  # , masters: Optional[List] = None):
    for dir_name, data, count in zip(["whole", "first 1k"], book_data, counts):  # , masters):
        for name in ALL_MODELS:
            if "gram" not in name:
                prep_for_df(data[name], count, name)
                data[name]["@Outcome"] = "tbd"

                if not isinstance(data[name], dict):
                    print("issue")
                    logging.error(f"Book # {book_number} -- data is not dictionary")

                # master[name].append(data[name])

                # row_dict = {"@Model": name}
                # row_dict.update(data[name])
                # row = [row_dict]
                row = [data[name]]
                with open(str(path.joinpath(dir_name, f"{book_number}_{name}_data")), "wb+") as f:
                    try:
                        row_df = pd.DataFrame(row).fillna(0)
                        pickle.dump(row_df, f)
                    except MemoryError:
                        print(f"There was a MemoryError when dumping ./{dir_name}/{book_number}_{name}_data")
                        logging.error(f"Book # {book_number} -- MemoryError dumping {dir_name} {name} data", exc_info=True)


def child_names(tree: Union[Tree, ParentedTree]):
    names = []
    num_pat = re.compile("[A-Za-z]*\\d+[A-Za-z]*", re.IGNORECASE)
    for child in tree:
        if isinstance(child, Tree):
            label = child.label()
        else:
            label = child
        if label not in string.punctuation and not re.match(num_pat, label):
            names.append(label)
    return '|'.join(names)


def productions(tree: Union[Tree, ParentedTree]):
    try:
        gnode = tree.parent().parent().label() + " -> "
    except AttributeError:
        gnode = "ROOT -> "

    prod = Production(Nonterminal(tree.label()), _child_names(tree))
    prodg = f"{gnode}{tree.label()} -> {child_names(tree)}"
    prods = {"lex": [], "lexg": [], "nonlex": [], "nonlexg": []}

    if prod.is_lexical():
        prods["lex"].append(prod)
        prods["lexg"].append(prodg)
    elif prod.is_nonlexical():
        prods["nonlex"].append(prod)
        prods["nonlexg"].append(prodg)

    for child in tree:
        if isinstance(child, Tree):
            c_prods = productions(child)
            for k, v in prods.items():
                if len(c_prods[k]) > 0:
                    prods[k] += c_prods[k]

    return prods


def get_rule_rhs(rule: Production):
    num_pat = re.compile("[A-Za-z]*\\d+[A-Za-z]*", re.IGNORECASE)
    rule_rhs = [str(rhs) for rhs in rule._rhs if str(rhs) not in string.punctuation and not re.match(num_pat, str(rhs))]
    return rule_rhs


def traverse_tree(tree: Union[Tree, ParentedTree], sent_data: Dict):
    if not isinstance(tree, ParentedTree):
        ptree = ParentedTree.convert(tree)
    else:
        ptree = tree

    prods = productions(ptree)
    for rule, rule_g in zip(prods["lex"], prods["lexg"]):
        rule_rhs = get_rule_rhs(rule)
        if len(rule_rhs) > 0:
            rule_str = f"{rule._lhs} -> {'|'.join(rule_rhs)}"
            word = str(rule._rhs[0]).lower()
            pos = str(rule._lhs)
            wn_pos = get_wn_pos(pos)

            for rocat in roget_thesaurus.get_categories(word):
                sent_data["roget"][rocat] += 1

            for emotion in NRC[word]:
                sent_data["nrc"][emotion] += 1

            for liwc_key in get_liwc_category(ps.stem(word)):
                sent_data["liwc"][liwc_key] += 1

            if wn_pos is not None:
                for wn_word in get_wn_words(word):
                    sent_data["wordnet"][wn_word] += 1

            sent_data["pos"][pos] += 1
            sent_data["lex"][rule_str] += 1
            sent_data["lexg"][rule_g] += 1

    for rule, rule_g in zip(prods["nonlex"], prods["nonlexg"]):
        rule_rhs = get_rule_rhs(rule)
        if len(rule_rhs) > 0:
            rule_str = f"{rule._lhs} -> {'|'.join(rule_rhs)}"
            if str(rule._lhs) in clausal:
                sent_data["clausal"][str(rule._lhs)] += 1

            elif str(rule._lhs) in phrasal:
                sent_data["phrasal"][str(rule._lhs)] += 1

            sent_data["nonlex"][rule_str] += 1
            sent_data["nonlexg"][rule_g] += 1


# region Old Traversal Method
def traverse_old(tree: Union[Tree, ParentedTree], sent_data: Dict, seen: set):
    if not isinstance(tree, ParentedTree):
        ptree = ParentedTree.convert(tree)
    else:
        ptree = tree

    for subtree in ptree:
        if isinstance(subtree, ParentedTree) and subtree.height() > 2:
            traverse_old(subtree, sent_data, seen)

    productions = ptree.productions()
    for rule in productions:
        if rule not in seen and not any(str(rhs) in string.punctuation for rhs in rule._rhs) and\
                not re.match("[A-Za-z]*\\d+[A-Za-z]*", str(rule._rhs), re.IGNORECASE):

            seen.add(rule)
            rule_str = f"{rule._lhs} -> {'|'.join([str(rhs) for rhs in rule._rhs])}"

            try:
                gnode = ptree.parent().parent().label() + " -> "
            except AttributeError:
                gnode = "ROOT -> "

            if rule.is_lexical():
                word = str(rule._rhs[0]).lower()
                pos = str(rule._lhs)
                wn_pos = get_wn_pos(pos)

                for rocat in roget_thesaurus.get_categories(word):
                    sent_data["roget"][rocat] += 1

                for emotion in NRC[word]:
                    sent_data["nrc"][emotion] += 1

                for liwc_key in get_liwc_category(ps.stem(word)):
                    sent_data["liwc"][liwc_key] += 1

                if wn_pos is not None:
                    for wn_word in get_wn_words(word):
                        sent_data["wordnet"][wn_word] += 1

                sent_data["pos"][pos] += 1
                sent_data["lex"][rule_str] += 1
                sent_data["lexg"][gnode + rule_str] += 1

            elif rule.is_nonlexical():
                if str(rule._lhs) in clausal:
                    sent_data["clausal"][str(rule._lhs)] += 1

                elif str(rule._lhs) in phrasal:
                    sent_data["phrasal"][str(rule._lhs)] += 1

                sent_data["nonlex"][rule_str] += 1
                sent_data["nonlexg"][gnode + rule_str] += 1
# endregion


def get_book_data(parser: CoreNLPParser, dl: DataLoader, books: pd.DataFrame, chunk: int, chunks: int, bar_props: tuple):
    logging.basicConfig(filename=PROJ_ROOT.joinpath(f"scripts/logs/get_book_data{chunk}.log"), filemode="w",
                        format="%(asctime)s::%(levelname)s - %(message)s", level=logging.DEBUG)
    logger = logging.getLogger()

    data_path = PROJ_ROOT.joinpath("data", "all book data")
    # make_data_dirs(data_path)

    bar_length, num_books = bar_props

    with tqdm(total=bar_length, position=chunk) as pbar:
        for i, (book_number, sentences) in enumerate(dl.all_text(books)):
            pbar.set_postfix_str(f"-- MINING -- {chunk + 1}/{chunks} -- [{i + 1}/{num_books}] ")
            logging.info(f"Book # {book_number} -- Analyzing book {i + 1} of {num_books}")

            # if book_number in finished_books:
            #     logging.info(f"Book # {book_number} -- Already processed, skipping")
            #     pbar.update(len(sentences))
            #     continue

            if len(sentences) < 50:
                logging.info(f"Book # {book_number} -- Less than 50 sentences")
                pbar.update(len(sentences))
                continue

            if all(len(s) == 1 for s in sentences[1].translate(remove_punct).split()):
                logging.info(f"Book # {book_number} -- b a d p a r s i n g")
                pbar.update(len(sentences))
                continue

            book_data, first_1k_data, j = mine_sentence(parser, book_number, sentences, pbar, logger)

            counts = get_counts(book_data)
            first_1k_counts = get_counts(first_1k_data)

            whole_book = {name: {"Book #": book_number, "@Genre": "tbd"} for name in NON_NGRAM}
            first_1k = {name: {"Book #": book_number, "@Genre": "tbd"} for name in NON_NGRAM}

            write_book_data(data_path, book_number, [whole_book, first_1k], [counts, first_1k_counts])

            if os.path.exists(str(PROJ_ROOT.joinpath("data", "all book data", "extras", f"processed_{chunk}"))):
                with open(str(PROJ_ROOT.joinpath("data", "all book data", "extras", f"processed_{chunk}")), "a") as f:
                    f.write(f"{book_number}\n")

            else:
                with open(str(PROJ_ROOT.joinpath("data", "all book data", "extras", f"processed_{chunk}")), "w+") as f:
                    f.write(f"{book_number}\n")

    print(f"--------- {chunk + 1}/{chunks} COMPLETE ---------")
    logging.info(f"--------- {chunk + 1}/{chunks} COMPLETE ---------")

        # for name in ALL_MODELS:
        #     if "gram" not in name:
        #         with open(str(PROJ_ROOT.joinpath("data", genre, f"{genre}{half}_{name}_data")), "wb+") as f:
        #             try:
        #                 genre_model_df = pd.DataFrame(genre_data[name]).fillna(0)
        #                 pickle.dump(genre_model_df, f)
        #             except MemoryError:
        #                 print(f"There was a MemoryError when dumping {genre}{half}_{name}_data")
        #                 logging.error(f"MemoryError dumping {genre} full books {name} data", exc_info=True)
        #         with open(str(PROJ_ROOT.joinpath("data", genre, f"{genre}{half}_1k_{name}_data")), "wb+") as f:
        #             try:
        #                 genre_model_df = pd.DataFrame(genre_data_1k[name]).fillna(0)
        #                 pickle.dump(genre_model_df, f)
        #             except MemoryError:
        #                 print(f"There was a MemoryError when dumping {genre}{half}_1k_{name}_data")
        #                 logging.error(f"MemoryError dumping {genre} 1k {name} data", exc_info=True)


def mine_sentence(parser: CoreNLPParser, book_number: str, sentences: List[str], pbar: tqdm, logger: logging.Logger):
    book_data = []
    first_1k_data = []
    i = 0

    for i, sentence in enumerate(sentences):
        if len(nltk.sent_tokenize(sentence.strip("\n"))) > 100:
            pbar.update(1)
            continue
        if all(re.match("\\d+", s) for s in sentence.split()):
            pbar.update(1)
            continue

        try:
            results = [r for r in parser.raw_parse(sentence.strip("\n"), properties={"annotators": "tokenize,ssplit,pos,parse",
                                                                                     "parse.maxlen": 100,
                                                                                     "threads": 1024,
                                                                                     "timeout": 15000})]

        except (AssertionError, RuntimeError, HTTPError) as e:
            logger.info(f"Book # {book_number} -- Error getting data from sentence: {sentence}")
            pbar.update(1)
            continue

        sent_data = {name: defaultdict(lambda: 0) if name != "nrc" else {emotion: 0 for emotion in EMOTIONS}
                     for name in ALL_MODELS if "gram" not in name}

        traverse_tree(results[0], sent_data)

        if i < 1000:
            first_1k_data.append(sent_data)
        book_data.append(sent_data)

        pbar.update(1)

    return book_data, first_1k_data, i


# if __name__ == "__main__":
#     argparser = ArgumentParser()
#     argparser.add_argument("--genre", help="The genre to mine")
#     argparser.add_argument("--half", help="Which half of the genre to mine [1, 2]")
#
#     args = argparser.parse_args()
#     text = load_all_text()
#     get_book_data(args.genre, args.half,,
