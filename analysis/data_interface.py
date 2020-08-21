import os
from pathlib import Path
from typing import List, Dict

from notebook_utils.constants import DATA_PATHS


class DataInterface:
    def __init__(self, jupyter: bool = False):
        self.jupyter = jupyter
        if self.jupyter:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        self.tqdm = tqdm

        self.data_paths: Dict[str, Path] = DATA_PATHS


class Loader(DataInterface):
    def __init__(self, first_1k: bool = False, jupyter: bool = False, **kwargs):
        super().__init__(jupyter)
        self.first_1k: bool = first_1k
        self.load_folder = "first 1k" if self.first_1k else "whole"
        self.save_folder = "first 1k" if self.first_1k else "whole"
        self.exclude: List[str] = ["30282", "42713", "12472", "12905", "36020"]
        if kwargs.get("exclude", None) is not None:
            self.exclude += kwargs.get("exclude")

    def load_all_text(self) -> List:
        all_books = []
        postfix = "-- LOADING TEXT"
        for root, dirs, files in os.walk(str(self.data_paths["text path"])):
            for file in self.tqdm(files, postfix=postfix):
                if file.endswith(".txt"):
                    book_number = file.split(".")[0]
                    if book_number not in self.exclude:
                        try:
                            text = open(str(self.data_paths["text path"].joinpath(file)), "r+").readlines()
                        except OSError:
                            continue
                        if len(text) > 5:
                            all_books.append((book_number, text[:1000] if self.first_1k else text))
        return all_books


# class Analyzer(DataInterface):
#     def __init__(self, model: str, first_1k: bool = False, by_genre: bool = False, jupyter: bool = False, **kwargs):
#         super().__init__(jupyter)
#         self.model: str = model
#         self.first_1k: bool = first_1k
#         self.by_genre: bool = by_genre
