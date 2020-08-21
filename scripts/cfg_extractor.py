import pickle
import string
from argparse import ArgumentParser
from collections import Counter, defaultdict

import pandas as pd
from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree
from tqdm import tqdm

from notebook_utils.constants import PROJ_ROOT, NEW_GENRES
from loading_utils.data_loader import load_all_books


def get_rules(tree, rules, seen: set):
    if not isinstance(tree, ParentedTree):
        ptree = ParentedTree.convert(tree)
    else:
        ptree = tree

    for subtree in ptree:
        if isinstance(subtree, ParentedTree) and subtree.height() > 2:
            get_rules(subtree, rules, seen)

    productions = ptree.productions()
    for rule in productions:
        if rule.is_lexical() and rule not in seen and str(rule._lhs) not in string.punctuation:
            rule_str = f"{rule._lhs} -> {' '.join([rhs for rhs in rule._rhs])}"
            rules["lex"][rule_str] += 1
            seen.add(rule)
            try:
                gnode = ptree.parent().parent().label() + " -> "
                rules["lexg"][gnode + rule_str] += 1
            except AttributeError:
                rules["lexg"]["ROOT -> " + rule_str] += 1
        elif rule.is_nonlexical() and rule not in seen and str(rule._lhs) not in string.punctuation:
            rules["nonlex"][str(rule)] += 1
            seen.add(rule)
            try:
                gnode = ptree.parent().parent().label() + " -> "
                rules["nonlexg"][gnode + str(rule)] += 1
            except AttributeError:
                rules["nonlexg"]["ROOT -> " + str(rule)] += 1


def get_cfg_rules(genre: str):
    all_books = load_all_books()
    all_rules = {genre: {"lex": [], "lexg": [], "nonlex": [], "nonlexg": []}}

    bar_length = len(all_books[genre]) * 1000

    parser = CoreNLPParser()

    with tqdm(total=bar_length, position=NEW_GENRES.index(genre), leave=True) as pbar:
        for i, book in enumerate(all_books[genre]):
            pbar.set_postfix_str(f" -- {genre} -- [{i + 1}/{len(all_books[genre])}] ")
            if book.book_number == "19513" or book.book_number == "19640" or book.book_number == "19678" \
                    or book.book_number == "19782" or book.book_number == "19836" or book.book_number == "22326" \
                    or book.book_number == "1322":
                pbar.update(1000)
                continue

            try:
                sentences = all_books[genre][i].first_1k_sentences
                book_rules = []

                for sentence in sentences:
                    results = [r for r in parser.raw_parse(sentence, properties={"annotators": "tokenize,ssplit,pos,parse"})]
                    sent_rules = {"lex": defaultdict(lambda: 0), "lexg": defaultdict(lambda: 0),
                                  "nonlex": defaultdict(lambda: 0), "nonlexg": defaultdict(lambda: 0)}
                    get_rules(results[0], sent_rules, set())

                    book_rules.append(sent_rules)

                    pbar.update(1)

                if len(all_books[genre][i].first_1k_sentences) < 1000:
                    pbar.update(1000 - len(all_books[genre][i].first_1k_sentences))

                counts = {"lex": sum([Counter(book_rules[j]["lex"]) for j in range(len(book_rules))], Counter()),
                          "lexg": sum([Counter(book_rules[j]["lexg"]) for j in range(len(book_rules))], Counter()),
                          "nonlex": sum([Counter(book_rules[j]["nonlex"]) for j in range(len(book_rules))], Counter()),
                          "nonlexg": sum([Counter(book_rules[j]["nonlexg"]) for j in range(len(book_rules))], Counter())}

                full_book_data = {"lex": {"Book #": all_books[genre][i].book_number, "@Genre": genre},
                                  "lexg": {"Book #": all_books[genre][i].book_number, "@Genre": genre},
                                  "nonlex": {"Book #": all_books[genre][i].book_number, "@Genre": genre},
                                  "nonlexg": {"Book #": all_books[genre][i].book_number, "@Genre": genre}}

                full_book_data["lex"].update({k: v for k, v in counts["lex"].items() if k != "''" and k != "``" and k not in string.punctuation})
                full_book_data["lex"]["@Outcome"] = all_books[genre][i].success

                full_book_data["lexg"].update({k: v for k, v in counts["lexg"].items() if k != "''" and k != "``" and k not in string.punctuation})
                full_book_data["lexg"]["@Outcome"] = all_books[genre][i].success

                full_book_data["nonlex"].update({k: v for k, v in counts["nonlex"].items() if k != "''" and k != "``" and k not in string.punctuation})
                full_book_data["nonlex"]["@Outcome"] = all_books[genre][i].success

                full_book_data["nonlexg"].update({k: v for k, v in counts["nonlexg"].items() if k != "''" and k != "``" and k not in string.punctuation})
                full_book_data["nonlexg"]["@Outcome"] = all_books[genre][i].success

                for tag_type in ["lex", "lexg", "nonlex", "nonlexg"]:
                    all_rules[genre][tag_type].append(full_book_data[tag_type])

            except (AssertionError, RuntimeError) as e:
                print(f"{genre}, {book.success}, {book.book_number}")
                pbar.update(1000)
                continue

        for tag_type in ["lex", "lexg", "nonlex", "nonlexg"]:
            with open(str(PROJ_ROOT.joinpath("data", genre, f"{genre}_{tag_type}_data")), "wb+") as f:
                try:
                    pickle.dump(pd.DataFrame(all_rules[genre][tag_type]).fillna(0), f)
                except MemoryError:
                    print(f"There was a MemoryError when dumping {genre}_{tag_type}_data")


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-genre", type=str, required=True, help="Genre to get CFG rules")

    args = p.parse_args()
    get_cfg_rules(args.genre)
