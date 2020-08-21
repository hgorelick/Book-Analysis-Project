__all__ = ['partition', 'partition_align_concat', 'drop_punctuation', 'partition_df', 'DataLoader', 'load_liwc', 'load_nrc']

import os
import pickle
import re
from collections import defaultdict
from typing import Dict, Optional, List
from zipfile import ZipFile

import numpy as np
import pandas as pd

from analysis.data_interface import Loader
from notebook_utils.constants import PROJ_ROOT


def partition(lst: List, length: int):
    for i in range(0, len(lst), length):
        yield lst[i:i + length]


def partition_align_concat(lst: List[pd.DataFrame]):
    result: List[pd.DataFrame] = []
    for df1, df2 in zip(lst[::2], lst[1::2]):
        if "model" in df1.columns:
            df1.drop(columns=["model"], inplace=True)
        if "model" in df2.columns:
            df2.drop(columns=["model"], inplace=True)
        df1, df2 = df1.align(df2, axis=1, fill_value=0)
        merged = pd.concat([df1, df2])
        merged = drop_punctuation(merged)
        result.append(merged)

    if len(lst) % 2 == 1:
        df1, df2 = result[-1].align(lst[-1], axis=1, fill_value=0)
        merged = pd.concat([df1, df2])
        merged = drop_punctuation(merged)
        result[-1] = merged

    return result


def drop_punctuation(df: pd.DataFrame):
    to_drop = []
    for col in df.columns:
        if col == "@Genre" or col == "Book #":
            break
        to_drop.append(col)
    df.drop(columns=to_drop, inplace=True)

    if any("->" in col for col in df.columns):
        df.drop(columns=[col for col in df.columns if "ADD" in col], inplace=True)
    try:
        df["Book #"] = df["Book #"].astype(str)
    except KeyError:
        df["Book #"] = "?"
    return df


def align_and_concat(df1: pd.DataFrame, df2):
    df1, df2 = df1.align(df2, axis=1, fill_value=0)
    return pd.concat([df1, df2])


def partition_df(df: pd.DataFrame, length: int, chunks: int, finished: Optional[List] = None, equal_avg: bool = False):
    if finished is not None:
        copy = df.loc[~df["Book #"].isin(finished)]
        length = len(copy) // chunks
    else:
        copy = df
    for i in range(0, len(copy), length):
        if equal_avg:
            yield copy.iloc[np.r_[(i // 2):(i + length) // 2, (len(copy) - 1) - ((i + length) // 2):(len(copy) - 1) - (i // 2)], :]
        else:
            yield copy.iloc[i:i + length, :]


class DataLoader(Loader):

    def __init__(self, first_1k: bool = False, jupyter: bool = False, exclude: Optional[List] = None):
        super().__init__(first_1k, jupyter, )

        self.book_details = self.load_book_info()
        self.init_mined = self.get_mined_list()

    @staticmethod
    def load_all_books() -> Dict:
        z = ZipFile(str(PROJ_ROOT.joinpath("data", "books_by_genre.zip")))
        namelist = z.namelist()

        return {re.search("(?<=all_).*(?=_books)", path)[0]: pickle.load(z.open(path)) for path in namelist}

    def get_init_mined(self):
        if os.path.exists(str(self.data_paths["processed list"])):
            processed_files = list(set(pickle.load(open(str(self.data_paths["processed list"]), "rb+"))))
            processed_files.sort()
            return processed_files

        processed_files = [file.rstrip("\n").split(" ")[-1]
                           for file in open(str(self.data_paths["all"].joinpath("processed_first1k.txt")), "r+").readlines()[1:]]
        processed_files = list(set(processed_files))
        processed_files.sort()

        with open(str(self.data_paths["processed list"]), "wb+") as f:
            pickle.dump(processed_files, f)

        return processed_files

    # def get_text(self, first_1k: bool = False):
    #     all_books = []
    #     postfix = "-- LOADING TEXT"
    #     for root, dirs, files in os.walk(str(self.data_paths["text path"])):
    #         for file in tqdm(files, postfix=postfix):
    #             if file.endswith(".txt"):
    #                 book_number = file.split(".")[0]
    #                 if book_number not in self.exclude:
    #                     try:
    #                         text = open(str(self.data_paths["text path"].joinpath(file)), "r+").readlines()
    #                     except OSError:
    #                         continue
    #                     if len(text) > 5:
    #                         all_books.append((book_number, text[:1000] if first_1k else text))
    #     return all_books

    def load_book_info(self):
        if os.path.exists(str(self.data_paths["all"].joinpath("book info"))):
            return pickle.load(open(str(self.data_paths["all"].joinpath("book info")), "rb+"))

        books_and_paths = self.books_and_paths()
        lengths = self.get_book_lengths()

        book_info = pd.merge(books_and_paths, lengths, how="outer", on="Book #")\
            .sort_values(by=["Length"], ascending=False)\
            .reset_index(drop=True)

        with open(str(self.data_paths["all"].joinpath("book info")), "wb+") as f:
            pickle.dump(book_info, f, protocol=4)

        return book_info

    def partition_books(self, chunks: int, finished_books: List):
        if chunks == 1:
            return self.book_details.loc[self.book_details["Book #"].isin(self.init_mined)].drop_duplicates()
        length = len(self.book_details) // 2
        return [chunk for chunk in partition_df(self.book_details, length, chunks, finished_books, True)]

    @staticmethod
    def all_text(books: pd.DataFrame):  # , finished_books: List, chunks: int = 0, chunks: int = 0):
        # books = self.books_and_paths(finished_books, chunks, chunks)
        for book_number, path in zip(books["Book #"], books["Path"]):
            try:
                yield book_number, open(path, "r+").readlines()
            except OSError:
                continue

    def books_and_paths(self, finished_books: Optional[List] = None, chunks: int = 0):
        book_numbers = []
        book_paths = []

        for root, dirs, files in os.walk(str(self.data_paths["text path"])):
            for file in files:
                if file.endswith(".txt"):
                    book_number = file.split(".")[0]
                    if book_number not in self.exclude:  # and book_number not in finished_books:
                        book_numbers.append(book_number)
                        book_paths.append(str(self.data_paths["text path"].joinpath(file)))

        books = pd.DataFrame({"Book #": book_numbers, "Path": book_paths})
        if chunks > 0:
            length = len(books) // chunks
            chunked = [c for c in partition_df(books, length, chunks, finished_books, True)]  # [chunks]
            return chunked

        return books

    def get_book_lengths(self) -> pd.DataFrame:
        if os.path.exists(str(self.data_paths["book lengths"])):
            book_lengths = self.mod_book_lengths()
            return book_lengths
        else:
            loaded = self.load_all_text()
            book_lengths = {"Book #": [], "Length": []}
            for book_number, sentences in self.tqdm(loaded, postfix=f"-- GETTING BOOK LENGTHS"):
                book_lengths["Book #"].append(book_number)
                book_lengths["Length"].append(len(sentences))

            book_lengths_df = pd.DataFrame(book_lengths).sort_values(by=["Length"], ascending=False)
            with open(str(self.data_paths["book lengths"]), "wb+") as f:
                pickle.dump(book_lengths, f)
            return book_lengths_df

    def mod_book_lengths(self) -> pd.DataFrame:
        modified = {"Book #": [], "Length": []}
        book_lengths = pickle.load(open(str(self.data_paths["book lengths"]), "rb+")).reset_index(drop=True)
        book_lengths = book_lengths.loc[~book_lengths["Book #"].isin(self.exclude)]
        if isinstance(book_lengths, pd.DataFrame):
            return book_lengths

        for genre, books in book_lengths.items():
            for b_num, length in books:
                modified["Book #"].append(b_num)
                modified["Length"].append(length)

        if len(modified["Book #"]) > 0:
            book_lengths_df = pd.DataFrame(modified).sort_values(by=["Length"], ascending=False)
            with open(str(self.data_paths["book lengths"]), "wb+") as f:
                pickle.dump(book_lengths_df, f)

            return book_lengths_df

        raise ValueError("length of Book #s is 0")

    def get_bar_props(self, chunks: int, finished_books: List, load: bool = False, save: bool = True):
        if load and os.path.exists(str(PROJ_ROOT.joinpath("data", f"bar-props-{chunks}"))):
            bar_props = pickle.load(open(str(PROJ_ROOT.joinpath("data", f"bar-props-{chunks}")), "rb+"))
            return bar_props

        elif chunks == 1:
            # length = len(self.book_details.loc[~self.book_details["Book #"].isin(self.init_mined)])
            length = len(self.init_mined)
            return length, length

        bar_props = []
        length = len(self.book_details) // chunks
        for chunk in partition_df(self.book_details, length, chunks, finished_books, equal_avg=True):
            bar_props.append((chunk["Length"].sum(), len(chunk)))

        if save:
            with open(str(PROJ_ROOT.joinpath("data", f"bar-props-{chunks}")), "wb+") as f:
                pickle.dump(bar_props, f)

        return bar_props

    def get_mined_list(self) -> List[str]:
        print("\n####### Checking for already processed files #######")
        processed_list = self.load_processed_list()
        if len(processed_list) > 0:
            return processed_list

        processed_list = self.create_processed_list()
        processed_list = list(set(processed_list))

        with open(str(self.data_paths["processed_list"]), "wb+") as f:
            pickle.dump(processed_list, f)

        return processed_list

    def load_processed_list(self) -> List[str]:
        processed_list = []
        if os.path.exists(str(self.data_paths["processed list"])):
            print("Found processed files, loading!")
            processed_list = pickle.load(open(str(self.data_paths["processed list"]), "rb+"))
            extras = []
            for extra in self.get_extras():
                processed_list = self.add_extra(processed_list, extra)
                extras += extra
        print("\n")
        return processed_list

    def get_extras(self):
        for file in self.tqdm(os.listdir(str(self.data_paths["all"].joinpath("extras"))), postfix="-- LOADING EXTRAS"):
            extra = [s.rstrip("\n") for s in open(str(self.data_paths["all"].joinpath("extras", file)), "r+").readlines()]
            yield extra

    def add_extra(self, processed: List, extra: List):
        processed += extra
        processed = self.set_and_sort(processed)
        return processed

    @staticmethod
    def set_and_sort(processed: List):
        processed = list(set(processed))
        processed.sort()
        return processed

    def create_processed_list(self):
        processed = []
        bad = []
        for file in self.tqdm(self.init_mined, postfix=f"-- GETTING PROCESSED"):
            b_num = file.split("_")[0]
            if b_num not in bad:
                file_path = self.data_paths["all"].joinpath("first 1k", file)
                try:
                    _ = pickle.load(open(str(file_path), "rb+"))
                except (EOFError, OSError, pickle.UnpicklingError):
                    bad.append(b_num)
                    while b_num in processed:
                        processed.remove(b_num)
                    continue
                processed.append(b_num)
        return processed

    # def lafs_bar_props(self):

    def load_all_for_save(self, name: str):
        # processed_1k = {name: [] for name in NON_NGRAM}
        # processed_whole = {name: [] for name in NON_NGRAM}

        bad = []
        for file in self.init_mined:  # tqdm(self.init_mined, postfix=f"-- LOADING ALL PROCESSED BOOKS"):
            b_num = file.split("_")[0]
            # for model in NON_NGRAM:
            file_name = f"{file}_{name}_data"
            if b_num not in bad:
                file_path = self.data_paths["all"].joinpath("first 1k", file_name)
                try:
                    data = pickle.load(open(str(file_path), "rb+"))
                    whole_path = re.sub("first 1k", "whole", str(file_path), re.IGNORECASE)
                    whole_data = pickle.load(open(str(whole_path), "rb+"))
                except (EOFError, OSError, pickle.UnpicklingError):
                    bad.append(b_num)
                    # for name, values in processed_1k.items():
                    #     for item in values:
                    #         if item["Book #"] == b_num:
                    #             processed_1k[name].remove(item)
                    #             processed_whole[name].remove(item)
                    continue
                # processed_1k[model].append(data)
                # processed_whole[model].append(whole_data)
                yield data, whole_data
        # return processed_1k, processed_whole

    # def load_ngram_data(self):
    #     ngram_data = []
    #     all_books = self.load_all_text()
    #
    #     # bar_length = sum(len(all_books[genre]) for genre, book in all_books.items())
    #
    #     # print("Loading ngram data..")
    #     with tqdm(total=len(all_books)) as pbar:
    #         # for genre, book in all_books.items():
    #         for i, (book_number, sentences) in enumerate(all_books):
    #             pbar.set_postfix_str(f"-- LOADING NGRAM DATA - [{i + 1}/{len(all_books)}]")
    #             if book_number in self.exclude or len(sentences) < 5 or all(len(s) == 1 for s in sentences[1].translate(remove_punct).split()):
    #                 pbar.update(1)
    #                 continue
    #
    #             text = " ".join(sentences)
    #             text = re.sub("_", "", text)
    #             text = re.sub("\n", "", text)
    #             text = text.translate(remove_punct)
    #             text = re.sub("chapter ([ivx]+\\s+|\\w+\\s+?)", "", text, re.IGNORECASE)
    #
    #             ngram_temp = {"Book #": book_number, "@Genre": "tbd", "text": text, "@Outcome": "tbd"}
    #             ngram_data.append(ngram_temp)
    #
    #             pbar.update(1)
    #
    #     ngram_df = pd.DataFrame(ngram_data)
    #     ngram_df.first_1k = ngram_df.first_1k.astype(str)
    #
    #     # with open(str(PROJ_ROOT.joinpath("data", "BOOK_NUMBERS")), "wb+") as f:
    #     #     try:
    #     #         pickle.dump(ngram_df["Book #"].reset_index(drop=True), f)
    #     #     except MemoryError:
    #     #         print(f"There was a MemoryError when dumping BOOK_NUMBERS")
    #
    #     return ngram_df, ngram_df.copy()


def load_liwc(ps):
    liwc_path = PROJ_ROOT.joinpath("data", "LIWC Data")
    liwc = {}
    for root, dirs, files in os.walk(str(liwc_path)):
        for file in files:
            if ".txt" in file:
                liwc.update({file.split(".")[0]:
                                 set([ps.stem(s.strip(" \n").strip("*").split("'")[0])
                                      for s in open(str(liwc_path.joinpath(file)), "r+").readlines()])})
    return liwc


def load_nrc():
    nrc_path = PROJ_ROOT.joinpath("data", "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
    nrc = defaultdict(lambda: [])
    nrc_file = open(str(nrc_path), "r+").readlines()

    for line in nrc_file:
        data = line.strip("\n").split("\t")
        word = data[0]
        emotion = data[1]
        score = int(data[2])
        if score > 0:
            nrc[word].append(emotion)
    return nrc


if __name__ == "__main__":
    dl = DataLoader()
    # dl.load_ngram_data()
