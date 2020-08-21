import os
import re
from pathlib import Path
from typing import List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

from book_processor.gutenberg_processor import process_rdf_df
from notebook_utils.constants import get_dir


def get_existing(path: Path):
    print("Getting already extracted files...")
    existing = []
    for r, ds, fs in os.walk(str(path)):
        for file in tqdm(fs):
            existing.append(file)
    return existing


def extract_htm(zf: ZipFile, path: Path, existing: List[str]):
    bad = ["12233-h", "jackind", "21765-h", "~", "26073-h", "8800-h", "Vol", "10020-h", "12472-h", "12905-h", "36020-h"]
    file_names = zf.namelist()
    for name in file_names:
        n = name.split("/")[-1]
        if not any(s in n for s in bad) and n not in existing and ".htm" in n and re.match("\\d+", n):
            with open(str(path.joinpath(n)), "wb") as new_f:
                new_f.write(zf.read(name))


def extract_txt(zf: ZipFile, path: Path, existing: List[str]):
    # bad = []
    file_names = zf.namelist()
    for name in file_names:
        n = name.split("/")[-1]
        if n not in existing and ".txt" in n and re.match("\\d+", n):
            with open(str(path.joinpath(n)), "wb") as new_f:
                new_f.write(zf.read(name))


def make_unzipped_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def unzip_files(path: Path, file_type: str):
    new_loc = "files" if file_type == "html" else "text files"
    zip_loc = "zipfiles" if file_type == "html" else "text zipfiles"
    ext = "-h" if file_type == "html" else ""

    exclude = [30282, 42713, 12472, 12905, 36020]

    make_unzipped_dir(str(path.joinpath(new_loc)))
    existing = get_existing(path.joinpath(new_loc))

    matches = []
    pattern = re.compile(f"\\d+{ext}\\.zip", re.IGNORECASE)
    print("Extracting new books...")
    for root, dirs, files in os.walk(str(path.joinpath(zip_loc))):
        book_files = np.array([int(bfile.split(f"{ext}.")[0]) for bfile in files if re.match(pattern, bfile)])
        book_files.sort()

        with tqdm(total=len(book_files)) as pbar:
            for b_num in book_files:
                if b_num in book_numbers and b_num not in exclude:
                    title = rdf_data_df.loc[rdf_data_df["Book #"] == b_num, "Title"].values[0]
                    pbar.set_postfix_str(f" -- [{b_num}, {title}")
                    with ZipFile(str(path.joinpath(zip_loc, f"{b_num}{ext}.zip")), "r") as z:
                        if file_type == "html":
                            extract_htm(z, path.joinpath(new_loc), existing)
                        else:
                            extract_txt(z, path.joinpath(new_loc), existing)
                    matches.append(b_num)
                pbar.update(1)

    # missing = np.setdiff1d(book_numbers, matches)
    # print(missing)


if __name__ == "__main__":
    rdf_data_df, by_genre_dict = process_rdf_df()

    book_numbers = rdf_data_df["Book #"].unique()
    book_numbers.sort()

    all_genres = rdf_data_df["Genre"].unique()
    all_genres.sort()

    files_path = get_dir(Path().cwd(), "zipfileLinks.txt")
    unzip_files(files_path, "txt")
