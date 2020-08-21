import logging
import os
import sys


sys.path.append(os.getcwd())
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import pickle
from multiprocessing import set_start_method, get_context  # , freeze_support
# from multiprocessing.pool import RemoteTraceback
from typing import List, Tuple

import pandas as pd
from requests.exceptions import HTTPError
from tqdm import tqdm
from nltk.parse import CoreNLPParser

from loading_utils.data_loader import DataLoader
from book_processor.gutenberg_processor import process_rdf_df
from notebook_utils.constants import NEW_GENRES, NON_NGRAM, PROJ_ROOT
from scripts.book_data_extractor import get_book_data


def make_model_dirs():
    for folder in ["first 1k", "whole"]:
        for name in NON_NGRAM:
            if not os.path.exists(str(PROJ_ROOT.joinpath("data", "all book data", "chunks", folder, name))):
                os.mkdir(str(PROJ_ROOT.joinpath("data", "all book data", "chunks", folder, name)))


def write_data(skip_processed: bool = False):
    bar_length = len(NON_NGRAM) * 2
    with tqdm(total=bar_length) as pbar:
        data_path = PROJ_ROOT.joinpath("data", "all book data")

        dl = DataLoader()
        finished_1k, finished_books = dl.load_all_for_save()

        if skip_processed:
            processed = [item for item in os.listdir(data_path.joinpath("chunks", "whole"))]
            if len(processed) == len(NON_NGRAM):
                pbar.update(bar_length)
                return

            else:
                unprocessed = list(set(NON_NGRAM).difference(set(processed)))
                unprocessed.sort()

        else:
            unprocessed = NON_NGRAM

        for name in unprocessed:
            pbar.set_postfix_str(f"-- CONCATENATING -- whole {name}")  # - {i + 1}/{chunks}")
            # chunk_data = [record for book in finished_books[name] for record in book.to_dict("records")]
            # data = pd.DataFrame(chunk_data)
            data = pd.concat(finished_books[name])

            pbar.set_postfix_str(f"-- CONCATENATING -- 1k {name}")  # - {i + 1}/{chunks}")
            # chunk_data_1k = [record for book in finished_1k[name] for record in book.to_dict("records")]
            # data_1k = pd.DataFrame(chunk_data_1k)
            data_1k = pd.concat(finished_1k[name])

            pbar.set_postfix_str(f"-- DUMPING -- whole {name}")  # - {i + 1}/{chunks}")
            with open(data_path.joinpath("chunks", "whole", f"{name}"), "wb+") as f:
                pickle.dump(data, f, protocol=4)
            pbar.update(1)

            pbar.set_postfix_str(f"-- DUMPING -- 1k {name}")  # - {i + 1}/{chunks}")
            with open(data_path.joinpath("chunks", "first 1k", f"{name}"), "wb+") as f:
                pickle.dump(data_1k, f, protocol=4)
            pbar.update(1)


def add_genres():
    data_path = PROJ_ROOT.joinpath("data", "all book data")
    # finished_1k = {name: [] for name in NON_NGRAM}
    # finished_whole = {name: [] for name in NON_NGRAM}

    rdf_data = process_rdf_df().sort_values(by=["Book #"]).reset_index(drop=True)

    dl = DataLoader()
    # finished_1k, finished_books = dl.load_all_for_save()
    with tqdm(total=len(NON_NGRAM) * 18154) as pbar:
        for name in NON_NGRAM:
            finished_1k = []
            finished_whole = []
            pbar.set_postfix_str(f"-- ADDING GENRES TO {name}")
            for first_1k, whole in dl.load_all_for_save(name):
                try:
                    first_1k["@Genre"] = rdf_data.loc[rdf_data["Book #"] == int(first_1k["Book #"].values[0]), "@Genre"].values[0]
                except IndexError:
                    pbar.update(1)
                    continue
                finished_1k.append(first_1k)
                whole["@Genre"] = rdf_data.loc[rdf_data["Book #"] == int(whole["Book #"].values[0]), "@Genre"].values[0]
                finished_whole.append(whole)
                pbar.update(1)

            print(f"DUMPING whole -- {name}")
            with open(data_path.joinpath("chunks", "whole", f"{name}"), "wb+") as f:
                pickle.dump(finished_whole, f, protocol=4)

            print(f"DUMPING 1k -- {name}")
            with open(data_path.joinpath("chunks", "first 1k", f"{name}"), "wb+") as f:
                pickle.dump(finished_1k, f, protocol=4)


# def write_data_pool(path: Path, genre: str):
#     make_data_dirs(path.joinpath(genre))
#     with tqdm(total=len(NON_NGRAM) - 2, position=NEW_GENRES.index(genre)) as pbar:
#         # for name in NON_NGRAM:
#         #     if "gram" not in name:
#         # for genre in NEW_GENRES:
#         finished_books, finished_1k = get_mined_books(path, by_model=True, which="both")
#
#         for name in NON_NGRAM:
#             if "gram" not in name:
#                 pbar.set_postfix_str(f" -- LOADING -- {name} - {genre}")
#                 genre_data = [record for num, book in finished_books[name].values() for record in book.to_dict("records")]
#                 genre_data_1k = [record for num, book in finished_1k[name].values() for record in book.to_dict("records")]
#
#                 # half1 = pickle.load(open(PROJ_ROOT.joinpath("data", genre, f"{genre}1_{name}_data"), "rb+"))
#                 # half2 = pickle.load(open(PROJ_ROOT.joinpath("data", genre, f"{genre}2_{name}_data"), "rb+"))
#                 #
#                 # half1_1k = pickle.load(open(PROJ_ROOT.joinpath("data", genre, f"{genre}1_1k_{name}_data"), "rb+"))
#                 # half2_1k = pickle.load(open(PROJ_ROOT.joinpath("data", genre, f"{genre}2_1k_{name}_data"), "rb+"))
#
#                 pbar.set_postfix_str(f" -- CONCATENATING -- {name} - {genre}")
#                 data = pd.concat(genre_data)
#                 data_1k = pd.concat(genre_data_1k)
#                 # data = pd.concat([half1, half2])
#                 # data1k = pd.concat([half1_1k, half2_1k])
#
#                 pbar.set_postfix_str(f" -- DUMPING -- {name} - {genre}")
#                 with open(PROJ_ROOT.joinpath("data", genre, f"{genre}_{name}_data"), "wb+") as f:
#                     pickle.dump(data, f)
#                 with open(PROJ_ROOT.joinpath("data", genre, f"{genre}1k_{name}_data"), "wb+") as f:
#                     pickle.dump(data_1k, f)
#                 pbar.update(1)


# def pool_write():
#     set_start_method("spawn")
#     args = [(PROJ_ROOT.joinpath("data"), genre) for genre in NEW_GENRES]
#     with get_context("spawn").Pool(processes=len(args)) as p:
#         p.starmap(write_data_pool, args)

def spawn_pool(args: List[Tuple], method: object):
    while True:
        try:
            with get_context("spawn").Pool(processes=len(args)) as p:
                p.starmap(method, args)
            p.close()
            p.join()
            return  # completed
        except (HTTPError, ConnectionError) as e:
            print(f"\n################ Pool threw requests error: {e.response} ################\n")
            logging.error(e.response)
            p.close()
            p.join()
        finally:
            p.close()
            p.join()


def pool_extract(chunks: int = 20, load: bool = False):
    logging.basicConfig(filename=PROJ_ROOT.joinpath("scripts/logs/pool_extract.log"), filemode="w",
                        format="%(asctime)s::%(levelname)s - %(message)s", level=logging.DEBUG)
    set_start_method("spawn")

    dl = DataLoader()
    # finished_books = dl.get_mined_list()
    bar_props = dl.get_bar_props(chunks=chunks, finished_books=dl.init_mined, load=load)
    books = dl.partition_books(chunks, dl.init_mined)

    parser = CoreNLPParser()

    args = [(parser, dl, books[chunk], chunk, len(bar_props), bar_props[chunk]) for chunk in range(len(bar_props))]

    return spawn_pool(args, get_book_data)


def get_unfinished_chunks(chunks: int):
    unfinished = {genre: [] for genre in NEW_GENRES}
    for genre in NEW_GENRES:
        for i in range(chunks):
            if os.path.exists(str(PROJ_ROOT.joinpath("scripts", "logs", f"{genre}{i}_data_extractor.log"))):
                with open(str(PROJ_ROOT.joinpath("scripts", "logs", f"{genre}{i}_data_extractor.log")), "rb+") as f:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                    last_line = f.readline().decode()
                    if "COMPLETE" not in last_line:
                        unfinished[genre].append(i)
    return unfinished


# def extract_by_genre(chunks: int = 24):
#     logging.basicConfig(filename=PROJ_ROOT.joinpath(f"scripts/logs/extract_by_genre.log"), filemode="w",
#                         format="%(asctime)s::%(levelname)s - %(message)s", level=logging.INFO)
#     set_start_method("spawn")
#
#     data_path = PROJ_ROOT.joinpath("data")
#     finished_books = get_mined_books(data_path)
#
#     for genre in NEW_GENRES:
#         bar_props = get_bar_props(chunks=chunks, finished_books=None)
#         print(f"\n------ MINING {genre} ------\n")
#         args = [(genre, i, chunks, finished_books[genre], bar_props[i][genre]) for i in range(chunks)]
#         spawn_pool(args)
#         print(f"\n------ {genre} COMPLETE ------\n")


# def test_pool_extract():
#     set_start_method("spawn")
#     data_path = PROJ_ROOT.joinpath("data")
#     finished_books = get_mined_books(data_path)
#
#     bar_props = get_bar_props("Horror", None)
#
#     args = [("Horror", i, finished_books["Horror"], bar_props[i]["Horror"]) for i in range(4)]
#     # pool1 = [(genre, 1, finished_books[genre], bar_props[1][genre]) for genre in NEW_GENRES]
#     # pool2 = [(genre, 2, finished_books[genre], bar_props[2][genre]) for genre in NEW_GENRES]
#     # args = pool1 + pool2
#     with get_context("spawn").Pool(processes=len(args)) as p:
#         p.starmap(get_book_data, args)


# def subproc_extract():
#     logging.basicConfig(filename=f"scripts/logs/subproc_extract.log", filemode="w",
#                         format="%(asctime)s::%(levelname)s - %(message)s", level=logging.INFO)
#     # print(os.getcwd())
#     for genre in NEW_GENRES:
#         h1 = subprocess.Popen(f"python ./scripts/book_data_extractor.py --genre {genre} --half 1",
#                               stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
#         h2 = subprocess.Popen(f"python ./scripts/book_data_extractor.py --genre {genre} --half 2",
#                               stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)


if __name__ == "__main__":
    # freeze_support()
    # subproc_extract()
    # pool_extract(64)
    # extract_by_genre()
    # test_pool_extract()
    # print("\n##### MINING COMPLETE #####\n")

    # write_data()
    add_genres()
