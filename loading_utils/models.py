__all__ = ['ModelLoader']

import os
import pickle
import re
import string
from typing import Dict, List, Optional, Union

import pandas as pd

from book_processor.gutenberg_processor import load_rdf_df
from loading_utils.data_loader import partition_align_concat, DataLoader
from analysis.data_interface import Loader
from notebook_utils.constants import CAPITALS, remove_punct
from notebook_utils.utils import process_and_scale, display_df, get_display_df


class ModelLoader(Loader):

    def __init__(self, model: str, first_1k: bool = False, by_genre: bool = False, jupyter: bool = False, **kwargs):
        super().__init__(first_1k, jupyter, **kwargs)

        self.mapping = kwargs.get("mapping", False)
        self.by_genre = by_genre
        if self.by_genre and not self.mapping:
            self.save_folder = f"by genre/{self.save_folder}"

        self.for_display: Dict = {}
        self.model: str = CAPITALS[model]
        self.data: Union[pd.DataFrame, List[pd.DataFrame]] = pd.DataFrame()

    def load_model(self):
        if "gram" in self.model.lower():
            self.load_ngrams()
            return
        folder = "not scaled" if self.mapping else "scaled"
        files = [file for file in os.listdir(str(self.data_paths["all"].joinpath(folder, self.save_folder)))
                 if self.model.lower() in file]
        if len(files) > 0:
            if "lex" in self.model.lower():
                print("Cannot load all lexical data at once, will load and train model in batches.")
                return
            print(f"Loading {self.model}...")
            count = 0
            for file in files:
                data = pickle.load(open(str(self.data_paths["all"].joinpath(folder, self.save_folder, file)), "rb+"))
                data = data.loc[data["Book #"] != 0]
                if count == 1:
                    self.data = [self.data]
                elif count > 1:
                    self.data.append(data)
                else:
                    self.data = data
                count += 1

        if len(self.data) > 0:
            print("Complete!")
            return

        with self.tqdm(total=5) as pbar:
            pbar.set_postfix_str(f"-- LOADING -- {self.model}")
            model_data = self.load_model_data(pbar)

            if isinstance(model_data, list):
                self.data = []
                with self.tqdm(total=len(model_data) * 2) as pbar2:
                    for i, data in enumerate(model_data):
                        self.scale_and_save(data, pbar2, str(i + 1))
                pbar.update(2)
            elif not self.mapping:
                self.scale_and_save(model_data, pbar)
            else:
                self.data = model_data
                self.for_display[self.model] = get_display_df(self.data)
                pbar.set_postfix_str(f"-- SAVING -- {self.model}")
                self.save(model_data, "not scaled")
                pbar.update(2)

    def load_ngrams(self):
        rdf_data = load_rdf_df()
        if os.path.exists(str(self.data_paths["all"].joinpath("chunks", "ngram_df"))):
            print(f"Loading {self.model}...")
            ngram_df = pickle.load(open(str(self.data_paths["all"].joinpath("chunks", "ngram_df")), "rb+"))
            print("Complete!")
            self.data = ngram_df
            return

        ngram_data = []
        dl = DataLoader(self.first_1k, self.jupyter, self.exclude)
        books = dl.partition_books(1, dl.init_mined)

        with self.tqdm(total=len(books)) as pbar:
            for i, (book_number, sentences) in enumerate(dl.all_text(books)):
                pbar.set_postfix_str(f"-- LOADING NGRAM DATA - [{i + 1}/{len(books)}]")
                if len(sentences) < 50 or all(len(s) == 1 for s in sentences[1].translate(remove_punct).split()):
                    pbar.update(1)
                    continue

                text = " ".join(sentences)
                text = re.sub("_", "", text)
                text = re.sub("\n", "", text)
                text = text.translate(remove_punct)

                try:
                    genre = rdf_data.loc[rdf_data["Book #"] == int(book_number), "@Genre"].values[0]
                except IndexError:
                    continue
                ngram_temp = {"Book #": book_number, "@Genre": genre, "text": text, "@Outcome": "tbd"}
                ngram_data.append(ngram_temp)

                pbar.update(1)

        ngram_df = pd.DataFrame(ngram_data)
        ngram_df.text = ngram_df.text.astype(str)

        self.data = ngram_df

        with open(str(self.data_paths["all"].joinpath("chunks", "ngram_df")), "wb+") as f:
            pickle.dump(ngram_df, f, protocol=4)

    def load_lex(self):
        files = [file for file in os.listdir(str(self.data_paths["all"].joinpath("scaled", self.save_folder)))
                 if self.model.lower() in file]
        for file in files:
            data = pickle.load(open(str(self.data_paths["all"].joinpath("scaled", self.save_folder, file)), "rb+"))
            data = data.loc[data["Book #"] != 0]
            yield data

    # def load_general(self):
    #     print("Loading general models...")
    #     unsaved = self.get_unsaved_scaled(WORD_CHOICE)
    #     if len(unsaved) == 0:
    #         return
    #
    #     self.process_and_save(unsaved)
    #
    # def load_lexical(self):
    #     print("Loading lexical data...")
    #     unsaved = self.get_unsaved_scaled(PRODUCTIONS)
    #     if len(unsaved) == 0:
    #         return
    #
    #     self.process_and_save(unsaved)

    # def get_unsaved_scaled(self, models: List[str]):
    #     unsaved = []
    #     for model in models:
    #         if os.path.exists(str(self.data_paths["all"].joinpath("scaled", self.folder, f"{model}_scaled"))):
    #             data = pickle.load(open(str(self.data_paths["all"].joinpath("scaled", self.folder, f"{model}_scaled")), "rb+"))
    #             for_display = get_display_df(data) if len(data.columns) > 8 else data.copy()
    #             # self.__setattr__(f"{name}_scaled", df)
    #             self.data[model] = data
    #             self.for_display[CAPITALS[model]] = for_display
    #         else:
    #             unsaved.append(model)
    #     return unsaved

    # def process_and_save(self, unsaved: List[str]):
    #     # genres = [genre for genre in NEW_GENRES if self.exclude is not None and genre not in self.exclude]
    #     # bar_length = len(unsaved) * len(NEW_GENRES)
    #     with self.tqdm(total=len(unsaved) * 4) as pbar:
    #         for model in unsaved:
    #             data: List[pd.DataFrame] = []
    #             # for genre in NEW_GENRES:
    #             pbar.set_postfix_str(f"-- LOADING -- {model}")
    #             model_data = self.load_model_data(model)
    #             data.append(model_data)
    #             # self.__setattr__(f"{model}_data", data)
    #             pbar.update(1)
    #
    #             self.scale_and_save(data, model, pbar)
    #
    #     # lex_data = pickle.load(open(PROJ_ROOT.joinpath("data", "lexical data", "ALL_lex_data"), "rb+"))
    #     # lexg_data = pickle.load(open(PROJ_ROOT.joinpath("data", "lexical data", "ALL_lexg_data"), "rb+"))
    #     # nonlex_data = pickle.load(open(PROJ_ROOT.joinpath("data", "lexical data", "ALL_nonlex_data"), "rb+"))
    #     # nonlexg_data = pickle.load(open(PROJ_ROOT.joinpath("data", "lexical data", "ALL_nonlexg_data"), "rb+"))
    #
    #     # print("Scaling lexical...")
    #     # self.lex_scaled, lex_display = process_and_scale(lex_data)
    #     # print("Scaling lexicalG...")
    #     # self.lexg_scaled, lexg_display = process_and_scale(lexg_data)
    #     # print("Scaling nonlexical...")
    #     # self.nonlex_scaled, nonlex_display = process_and_scale(nonlex_data)
    #     # print("Scaling nonlexicalG...")
    #     # self.nonlexg_scaled, nonlexg_display = process_and_scale(nonlexg_data)
    #
    #     # self.for_display.update({"Lex": lex_display, "LexG": lexg_display, "Nonlex": nonlex_display, "NonlexG": nonlexg_display})

    def load_model_data(self, pbar: Optional = None):
        model_data = pickle.load(open(str(self.data_paths["all"].joinpath("chunks", self.load_folder, self.model.lower())), "rb+"))
        if isinstance(model_data, list):
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"-- CONCATENATING -- {self.model}")
            else:
                print(f"\nConcatenating {self.model}...")
        error = False
        while len(model_data) != 1:
            try:
                model_data = partition_align_concat(model_data)
            except MemoryError:
                error = True
                break
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix_str(f"-- CLEANING -- {self.model}")
        else:
            print(f"\nCleaning {self.model}...")
        if not error:
            model_data = model_data[0].sort_values(by=["@Genre"])
            clean = self.clean_data(model_data)
        else:
            clean = []
            for data in model_data:
                clean.append(self.clean_data(data))
        pbar.update(1)
        return clean

    def scale_and_save(self, data: pd.DataFrame, pbar, ext: str = ""):
        pbar.set_postfix_str(f"-- SCALING -- {self.model}")
        if self.by_genre:
            scaled, for_display = self.scale_by_genre(data)
        else:
            scaled, for_display = process_and_scale(data, self.model.lower())
        if ext != "":
            self.data.append(scaled)
        else:
            self.data = scaled
        self.for_display[self.model] = for_display
        pbar.update(1)

        pbar.set_postfix_str(f"-- SAVING -- {self.model}")
        self.save(scaled, "scaled", ext)
        pbar.update(1)

    def save(self, data: pd.DataFrame, folder: str, ext: str = ""):
        with open(str(self.data_paths["all"].joinpath(folder, self.save_folder, f"{self.model.lower()}_scaled{ext}")), "wb+") as f:
            pickle.dump(data, f, protocol=4)

    def scale_by_genre(self, data):
        groupby_genre = data.groupby(by=["@Genre"])
        scaled_by_genre = []
        scaled_display = []
        for genre, group in groupby_genre:
            s, fd = process_and_scale(group, self.model.lower())
            scaled_by_genre.append(s)
            scaled_display.append(fd)
        scaled = pd.concat(scaled_by_genre).reset_index(drop=True)
        for_display = pd.concat(scaled_display).reset_index(drop=True)
        return scaled, for_display

    @staticmethod
    def clean_data(model_data: pd.DataFrame):
        model_data = model_data.loc[:, ~model_data.columns.duplicated()]
        model_data["@Outcome"] = "tbd"
        # model_data.insert(0, "@Model", name)
        # if "model" in model_data.columns:
        #     model_data = model_data.drop(columns=["model"])
        special_cols = ["Book #", "@Genre", "@Outcome", "@Downloads"]
        model_data = model_data[["Book #", "@Genre"] + [col for col in model_data.columns if col not in special_cols] + ["@Outcome", "@Downloads"]]
        if any("->" in col for col in model_data.columns):
            to_drop = []
            for col in model_data.columns:
                for symbol in col.split(" -> "):
                    if symbol in string.punctuation:
                        to_drop.append(col)
                        break
            model_data.drop(columns=to_drop, inplace=True)
        if "ADD" in model_data.columns:
            model_data.drop(columns=["ADD"], inplace=True)
        return model_data.reset_index(drop=True)

    # def get_model(self):
        # if "gram" in model_name:
        #     return self.__getattribute__(f"{model_name.lower()}_df")
        # else:
        #     return self.__getattribute__(f"{model_name.lower()}_scaled")
        # return self.data

    def show_model_df(self, model_name: str):
        if "gram" in model_name:
            display_df(self.data, f"<h4>{model_name} Data</h4>", max_rows=6,
                       formatters={"text": lambda s: s[:100] + "..."})
        else:
            display_df(self.for_display[model_name], f"<h4>{model_name} Data</h4>", max_rows=10)
