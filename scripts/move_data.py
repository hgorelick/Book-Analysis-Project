import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import shutil
import sys
from multiprocessing import set_start_method, get_context  # , freeze_support
# from multiprocessing.pool import RemoteTraceback
from typing import List, Tuple
from itertools import product
from tqdm import tqdm

sys.path.append(os.getcwd())

from notebook_utils.constants import NEW_GENRES, PROJ_ROOT


def spawn_pool(args: List[Tuple]):
    while True:
        with get_context("spawn").Pool(processes=len(args)) as p:
            p.starmap(move_data_files, args)
        p.join()
        p.close()


def move_data_files(genre: str, folder: str):
    dest = PROJ_ROOT.joinpath("data", "all book data")
    src = PROJ_ROOT.joinpath("data", genre, folder)
    for file in tqdm(os.listdir(str(src)), postfix=f"-- MOVING DATA -- {folder} - {genre}"):
        shutil.move(str(src.joinpath(file)), str(dest.joinpath(folder, file)))


def move_all():
    dest = PROJ_ROOT.joinpath("data", "all book data")
    for genre in NEW_GENRES:
        for folder in ["first 1k", "whole"]:
            src = PROJ_ROOT.joinpath("data", genre, folder)
            for file in tqdm(os.listdir(str(src)), postfix=f"-- MOVING DATA -- {folder} - {genre}"):
                shutil.move(str(src.joinpath(file)), str(dest.joinpath(folder, file)))


def pool_move():
    set_start_method("spawn")
    args = [tuple(item) for item in product(*[NEW_GENRES, ["first 1k", "whole"]])]
    return spawn_pool(args)


if __name__ == "__main__":
    # pool_move()
    move_all()
