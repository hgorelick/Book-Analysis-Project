__all__ = ['Predictor']

import gc
import os
import pickle
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from IPython.core.display import display, HTML
from sklearn import preprocessing, svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.utils._testing import ignore_warnings

from analysis.data_interface import DataInterface
from book_processor.gutenberg_processor import load_rdf_df
from loading_utils.models import ModelLoader
from notebook_utils.constants import NEW_GENRES, GENRE_COMBS, NO_HORROR, scaler, CAPITALS, minmax
from notebook_utils.utils import display_df, filter_out_zeros


# TODO: Find average difference between predicted rank and actual rank


class Predictor(DataInterface):
    def __init__(self, model: str = "", first_1k: bool = False, by_genre: bool = False, jupyter: bool = False,
                 model_loader: Optional[ModelLoader] = None, estimator: str = "svm", **kwargs):
        self.estimator: str = estimator
        if model_loader is not None and not kwargs.get("mapping", False):
            self.model_loader = model_loader
            super().__init__(jupyter=self.model_loader.jupyter)
            self.model: str = self.model_loader.model
            self.first_1k: bool = self.model_loader.first_1k
            self.by_genre: bool = self.model_loader.by_genre
        else:
            super().__init__(jupyter=jupyter)
            self.model: str = CAPITALS[model]
            self.first_1k: bool = first_1k
            self.by_genre: bool = by_genre
            self.model_loader = ModelLoader(model, self.first_1k, self.by_genre, self.jupyter, **kwargs)
            self.model_loader.load_model()

            with_downloads = self.merge_with_downloads(rank=kwargs.get("rank", False))
            self.downloads = with_downloads[["Book #", "@Genre", "@Outcome", "@Downloads"]].sort_values(by=["@Outcome"], ascending=False)

            if "lr" in self.estimator:
                self.model_loader.data = with_downloads
                if kwargs.get("scale", True):
                    self.scale_outcome()

    def scale_outcome(self):
        if self.by_genre:
            for genre in NO_HORROR:
                self.model_loader.data.loc[self.model_loader.data["@Genre"] == genre, "@Outcome"] = scaler.fit_transform(
                    self.model_loader.data.loc[self.model_loader.data["@Genre"] == genre][["@Outcome"]])
        else:
            self.model_loader.data["@Outcome"] = scaler.fit_transform(self.model_loader.data[["@Outcome"]])

    def merge_with_downloads(self, df_: Optional[pd.DataFrame] = None, rank: bool = False, **kwargs):
        rdf_data = load_rdf_df()
        rdf_data["Book #"] = rdf_data["Book #"].astype(str)
        cols = ["Book #", "@Downloads"]
        cols += kwargs.get("extras", [])
        if df_ is not None:
            with_downloads = pd.merge(df_, rdf_data[cols], how="outer", on=["Book #"])
        else:
            with_downloads = pd.merge(self.model_loader.data, rdf_data[cols], how="outer", on=["Book #"])

        with_downloads["@Outcome"] = with_downloads["@Downloads"].copy()
        if rank:
            with_downloads = with_downloads.sort_values(by=["@Downloads"], ascending=False) \
                .reset_index() \
                .drop(columns=["@Outcome"]) \
                .rename(columns={"index": "@Outcome"})
        return with_downloads

    def predict_by_genre(self, genre_list: List[str] = NO_HORROR, **kwargs):
        """
        - add_to_acc: Union[List, Dict] = None
        - disp_acc = True
        - disp_weights = True
        - ratio: int = None
        - searching = False
        - show_pbar = True
        """
        accuracies = []
        weights = {genre: [] for genre in genre_list}
        preds = {genre: [] for genre in genre_list}

        if kwargs.get("show_pbar", True):
            display(HTML(f"<h4>Predicting book {'success' if self.estimator == 'svm' else 'downloads'} with {self.model} data...</h4>"))
            bar_length = len(genre_list) * 5
            with self.tqdm(total=bar_length) as pbar:
                self._predict_by_genre(accuracies, weights, preds, genre_list, pbar, **kwargs)
        else:
            self._predict_by_genre(accuracies, weights, preds, genre_list, **kwargs)

        accuracies = self.process_accuracies(pd.DataFrame(accuracies), **kwargs)
        weights = self.process_weights_by_genre(weights, disp=kwargs.get("disp_weights", True))

        return accuracies, weights, preds

    def _predict_by_genre(self, accs: List, ws: Dict, ps: Dict, genre_list: List[str] = NO_HORROR, pbar: Optional = None, **kwargs):
        """
        - searching = False
        :param ps:
        """
        for genre in genre_list:
            if pbar is not None:
                pbar.set_postfix_str(f" -- {genre}")

            mean_acc, preds = self._train_test(ws, genre, "@Outcome", pbar, **kwargs)

            accuracy = np.array(mean_acc).mean()
            accs.append({"Genre": genre, "Accuracy": accuracy})
            ps[genre] = preds

    # def predict_downloads_by_genre(self, genre_list: List[str] == NO_HORROR, **kwargs):
    #     accuracies = []
    #     weights = {genre: [] for genre in genre_list}
    #
    #     if kwargs.get("show_pbar", True):
    #         display(HTML(f"<h4>Predicting book downloads with {self.model} data...</h4>"))
    #         bar_length = len(genre_list) * 5
    #         with self.tqdm(total=bar_length) as pbar:

    @staticmethod
    def process_weights_by_genre(model_weights: Dict, disp: bool = True):
        for key, weights in model_weights.items():
            model_weights[key] = pd.concat(weights)
            model_weights[key].reset_index(drop=True, inplace=True)
            model_weights[key] = model_weights[key].mean(axis=0).reset_index()
            model_weights[key].columns = ["Feature", "Weight"]
            model_weights[key] = model_weights[key].sort_values(by=["Weight"], ascending=False).reset_index(drop=True)

            if disp:
                display_df(model_weights[key], f"<h4>{key} Feature Weights</h4>", 10, True)

        return model_weights

    def predict_all(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        - add_to_acc: Union[List, Dict] = None
        - disp_acc = True
        - disp_weights = True
        - ratio: int = None
        - searching = False
        - show_pbar = True
        """
        self.estimator = kwargs.get("estimator", self.estimator)
        accuracies = []
        weights = []

        # if kwargs.get("pbar", None) is not None:
        #     preds = self._predict_all(accuracies, weights, all=True, **kwargs)

        if kwargs.get("show_pbar", True) and kwargs.get("pbar", None) is None:
            display(HTML(f"<h4>Predicting book success with {self.model} data...</h4>"))
            bar_length = 5
            with self.tqdm(total=bar_length) as pbar:
                preds = self._predict_all(accuracies, weights, pbar, all=True, **kwargs)
        else:
            preds = self._predict_all(accuracies, weights, all=True, **kwargs)

        accuracies = self.process_accuracies(pd.DataFrame(accuracies), all=True, **kwargs)
        weights = self.process_weights(weights, disp=kwargs.get("disp_weights", False))

        return accuracies, weights, preds

    def _predict_all(self, accs: List, ws: List, pbar: Optional = None, **kwargs):
        """
        - searching = False
        """
        mean_acc, preds = self._train_test(ws, None, "@Outcome", pbar, **kwargs)

        accuracy = np.array(mean_acc).mean()
        accs.append({"Genre": "All", "Accuracy": accuracy})

        return preds

    @staticmethod
    def process_weights(model_weights: List, disp: bool = True) -> pd.DataFrame:
        processed_weights = pd.concat(model_weights)
        processed_weights.reset_index(drop=True, inplace=True)
        processed_weights = processed_weights.mean(axis=0).reset_index()
        processed_weights.columns = ["Feature", "Weight"]
        processed_weights = processed_weights.sort_values(by=["Weight"], ascending=False).reset_index(drop=True)

        if disp:
            display_df(processed_weights, f"<h4>Feature Weights</h4>", 10, True)

        return processed_weights

    def process_accuracies(self, accuracies: pd.DataFrame, **kwargs):
        if not kwargs.get("all", False):
            accuracies = accuracies.append({"Genre": "Average", "Accuracy": accuracies["Accuracy"].mean()}, ignore_index=True)
        if kwargs.get("add_to_acc", None) is not None:
            if isinstance(kwargs["add_to_acc"], list):
                kwargs["add_to_acc"].append(accuracies)
            if isinstance(kwargs["add_to_acc"], dict):
                kwargs["add_to_acc"].update({self.model: accuracies})
        if kwargs.get("disp_acc", True):
            display_df(accuracies, f"<h4>{self.model} Accuracies by Genre</h4>")
        return accuracies

    # @staticmethod
    # def process_results(results: pd.Series):
    #     if all(len(results[i]) > 0 for i in range(len(results))):
    #         processed_results = pd.concat(results)
    #         processed_results = processed_results.mean(axis=0).reset_index(drop=True)
    #     else:
    #         processed_results = pd.DataFrame()
    #     return processed_results

    def predict_genre(self, how: str = "one_v_one", genre_list: List = GENRE_COMBS, **kwargs):
        """
        - add_to_acc: Dict = None
        - disp_acc = True
        - disp_weights = True
        - searching = False
        - show_pbar = True
        """
        accuracies = []
        weights = {genre: [] for genre in genre_list}

        if kwargs.get("show_pbar", True):
            display((HTML(f"<h4>Performing {how} binary genre prediction with {self.model} data...</h4>"),))
            bar_length = len(genre_list) * 5
            with self.tqdm(total=bar_length) as pbar:
                self.__getattribute__(how)(accuracies, weights, genre_list, pbar, **kwargs)

        else:
            self.__getattribute__(how)(accuracies, weights, genre_list, **kwargs)

        accuracies = pd.DataFrame(accuracies)
        weights = self.process_weights_by_genre(weights, disp=kwargs.get("disp_weights", True))
        if kwargs.get("add_to_acc", None) is not None:
            kwargs["add_to_acc"].update({self.model: accuracies})

        accuracies = accuracies.append({"Genre": "Average", "Accuracy": accuracies["Accuracy"].mean()}, ignore_index=True)
        if kwargs.get("disp_acc", True):
            display_df(accuracies, f"<h4>{self.model} Accuracies by Genre</h4>")

        return accuracies, weights

    def one_v_one(self, accs: List, ws: Dict, genre_list: List, pbar: Optional = None, **kwargs):
        for g1, g2 in genre_list:
            if pbar is not None:
                pbar.set_postfix_str(f" -- {g1}, {g2}")

            mean_acc, _ = self._train_test(ws, (g1, g2), "@Genre", pbar, **kwargs)

            accuracy = np.array(mean_acc).mean()
            accs.append({"Genre": (g1, g2), "Accuracy": accuracy})

    def one_v_all(self, accs: List, ws: Dict, genre_list: List, pbar: Optional = None, **kwargs):
        for genre in genre_list:
            if pbar is not None:
                pbar.set_postfix_str(f" -- {genre}")

            gtemp = self.model_loader.data[self.model_loader.data["@Genre"] == genre].copy().reset_index(drop=True)
            not_gtemp = self.model_loader.data[self.model_loader.data["@Genre"] != genre].copy().reset_index(drop=True)

            sample_idx = random.sample(range(0, len(not_gtemp)), k=len(gtemp))

            not_gtemp = not_gtemp.iloc[sample_idx].reset_index(drop=True)
            not_gtemp["@Genre"] = f"not {genre}"

            df_temp = pd.concat([gtemp, not_gtemp])
            mean_acc, _ = self._train_test(ws, genre, "@Genre", pbar, temp=df_temp, **kwargs)

            accuracy = np.array(mean_acc).mean()
            accs.append({"Genre": genre, "Accuracy": accuracy})

    def _train_test(self, ws: Union[List, Dict], wkey: Optional[Union[str, tuple]], pred_col: str, pbar: Optional = None, **kwargs):
        tfi_data, y_data, temp, cols = self.get_tfi_and_y_data(pred_col, wkey, **kwargs)

        if kwargs.get("ratio", 0) > 0:
            cv = StratifiedShuffleSplit(n_splits=5, test_size=kwargs["ratio"], random_state=0)
            split = cv.split(tfi_data, y_data)
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=0)
            split = cv.split(y_data)

        mean_acc = []
        results = []
        for train_index, test_index in split:
            try:
                x_train, x_test = tfi_data.iloc[train_index], tfi_data.iloc[test_index]
            except AttributeError:
                x_train, x_test = tfi_data[train_index], tfi_data[test_index]
            try:
                y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
            except AttributeError:
                y_train, y_test = y_data[train_index], y_data[test_index]

            preds, clf = self.__predict(x_train, x_test, y_train)

            coefs = clf.coef_.ravel()
            if kwargs.get("all", False):
                try:
                    ws.append(pd.DataFrame(coefs, index=tfi_data.columns).transpose())
                except AttributeError:
                    ws.append(pd.DataFrame(coefs, index=cols))
            else:
                try:
                    ws[wkey].append(pd.DataFrame(coefs, index=tfi_data.columns).transpose())
                except AttributeError:
                    ws[wkey].append(pd.DataFrame(coefs, index=cols))

            if self.estimator == "svm":
                score = np.mean(preds == y_test)
            else:
                score = 1.0 - mean_squared_error(y_test, preds)
                results.append(pd.DataFrame({"@Outcome": preds}, index=list(y_test.index)))
            mean_acc.append(score)

            # if self.estimator == "lr rank":
            #     results.append(pd.Series(preds, index=y_test.index))

            if pbar is not None:
                pbar.update(1)

        if len(results) > 0:
            preds_df = pd.concat(results, axis=1)
            preds_df = preds_df.mean(axis=1).reset_index().sort_values(by=["index"])
            preds_df.columns = ["index", "Pred"]
            preds_df = pd.merge(temp[["Book #", "@Genre", "@Outcome", "@Downloads"]].reset_index().sort_values(by=["index"]), preds_df,
                                  how="outer", on="index").drop(columns="index").dropna()
        else:
            preds_df = pd.DataFrame()

        # print(temp[["Book #", "@Genre", "@Outcome"]])
        return mean_acc, preds_df

    def get_tfi_and_y_data(self, pred_col, wkey, **kwargs):
        if kwargs.get("temp", None) is not None:
            temp = kwargs["temp"]
            if wkey is not None:
                temp = Predictor.filter_df(temp, "@Genre", wkey)
        elif kwargs.get("all", False):
            temp = self.model_loader.data
        else:
            temp = self.filter_model_data("@Genre", wkey)

        cols = []
        if not kwargs.get("reducing", False):
            tfi_data, cols = self.get_df_by_name(temp)
        else:
            tfi_data = temp.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"])

        if self.estimator == "svm":
            encoder = preprocessing.LabelEncoder()
            y_data = encoder.fit_transform(temp[pred_col])
        else:
            y_data = temp[pred_col]

        return tfi_data, y_data, temp, cols

    def filter_model_data(self, filter_key: str, filter_val):
        if isinstance(filter_val, tuple):
            return self.model_loader.data[(self.model_loader.data[filter_key] == filter_val[0]) |
                                          (self.model_loader.data[filter_key] == filter_val[1])].reset_index(drop=True)
        else:
            return self.model_loader.data[self.model_loader.data[filter_key] == filter_val].copy().reset_index(drop=True)

    @staticmethod
    def filter_df(df: pd.DataFrame, filter_key: str, filter_val: str):
        return df[df[filter_key] == filter_val].copy()  # .reset_index(drop=True)

    def get_df_by_name(self, temp: pd.DataFrame):  # , searching: bool = False):
        if "gram" not in self.model:
            return temp.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]), []

        return self.tfi_ngram(temp)

    @ignore_warnings(category=ConvergenceWarning)
    def __predict(self, x_train, x_test, y_train):
        if self.estimator == "svm":
            est = svm.LinearSVC()
        elif "lr" in self.estimator:
            est = LinearRegression()
        else:
            raise RuntimeError("Bad argument given for model.")
        est.fit(x_train, y_train)
        preds = est.predict(x_test)
        return preds, est

    def tfi_ngram(self, df_temp: pd.DataFrame):
        if "uni" in self.model.lower():
            tfi_ngram_df_vect = CountVectorizer(analyzer="word")
        elif "bi" in self.model.lower():
            tfi_ngram_df_vect = CountVectorizer(analyzer="word", ngram_range=(2, 2))
        else:
            return None

        # tfi_ngram_df_vect.fit(df_temp["text"])
        tfi_ngram_data = tfi_ngram_df_vect.fit_transform(df_temp["text"])
        try:
            dense = tfi_ngram_data.todense()
            if self.by_genre:
                tfi_ngram_data = filter_out_zeros(dense, tfi_ngram_df_vect.get_feature_names())
            tfi_ngram_data = self.remove_numbers(tfi_ngram_data)
            tfi_ngram_data.insert(0, "@Genre", df_temp["@Genre"])
            tfi_ngram_data.insert(0, "Book #", df_temp["Book #"])
            tfi_ngram_data["@Outcome"] = df_temp["@Outcome"]
            tfi_ngram_data["@Downloads"] = df_temp["@Downloads"]
            return tfi_ngram_data, []
        except MemoryError:
            return tfi_ngram_data, tfi_ngram_df_vect.get_feature_names()

    @staticmethod
    def remove_numbers(df: pd.DataFrame):
        try:
            drop_cols = [c for c in df.columns if re.match("[A-Za-z]*\\d+[A-Za-z]*", c, re.IGNORECASE)]
            dropped = df.drop(columns=drop_cols)
            return dropped
        except AttributeError:
            return df

    @staticmethod
    def get_ens_str(model_names: List[str]):
        ens_str = []
        for name in model_names:
            if name.lower() == "lex":
                ens_str.append("$\Gamma$")
            elif name.lower() == "lexg":
                ens_str.append("$\Gamma^G$")
            elif name.lower() == "nonlex":
                ens_str.append("$\gamma$")
            elif name.lower() == "nonlexg":
                ens_str.append("$\gamma^G$")
            else:
                ens_str.append(name)
        return "_".join(ens_str)

    # TODO: NEEDS MAJOR REWORKING TO FIT NEW STRUCTURE --> Single model loading per predictor, no model name variable
    @ignore_warnings(category=ConvergenceWarning)
    def ensemble(self, model_names: List[str], **kwargs):
        """
        - add_to_acc: Union[List, Dict] = None
        - disp_acc = True
        - disp_weights = True
        - searching = False
        - show_pbar = True
        """
        ens_acc = []
        ens_str = self.get_ens_str(model_names)

        disp_str = ens_str.split("_")
        if len(disp_str) < 3:
            disp_str = " and ".join(disp_str)
        else:
            first_n = ', '.join(disp_str[:-1])
            disp_str = first_n + f", and {disp_str[-1]}"

        display(HTML(f"<h4>Performing ensemble book success prediction with {disp_str}...</h4>"))
        bar_length = len(NEW_GENRES) * 5

        with self.tqdm(total=bar_length) as pbar:
            for genre in NEW_GENRES:
                pbar.set_postfix_str(f" -- {genre}")
                encoder = preprocessing.LabelEncoder()
                models_ = []

                for name in model_names:
                    temp = self.filter_model_data("@Genre", genre)
                    tfi_data = self.get_df_by_name(temp)
                    y_data = encoder.fit_transform(temp["@Outcome"])
                    models_.append((tfi_data, y_data))

                kf = KFold(n_splits=5, shuffle=True, random_state=0)
                mean_acc = []

                for train_index, test_index in kf.split(models_[0][1]):
                    X_trains = []
                    X_tests = []
                    y_trains = []
                    y_tests = []

                    for x, y in models_:
                        X_trains.append(x.iloc[train_index])
                        X_tests.append(x.iloc[test_index])
                        y_trains.append(y[train_index])
                        y_tests.append(y[test_index])

                    preds = []
                    probs = []
                    for x_train, x_test, y_train, y_test in zip(X_trains, X_tests, y_trains, y_tests):

                        estimator = svm.LinearSVC()
                        estimator.fit(x_train, y_train)
                        preds.append(estimator.predict(x_test))

                        if len(models_) < 3:
                            probs.append(estimator._predict_proba_lr(x_test))

                    ens = []
                    if len(models_) > 2:
                        for pred in zip(*preds):
                            counter = Counter(pred)
                            ens.append(max(counter, key=counter.get))
                    else:
                        preds1, preds2 = preds[0], preds[1]
                        probs1, probs2 = probs[0], probs[1]
                        for pred1, prob1, pred2, prob2 in zip(preds1, probs1, preds2, probs2):
                            pred = round(((pred1 * prob1[1]) + (pred2 * prob2[1])) / 2)
                            ens.append(pred)

                    score = np.mean(ens == y_tests[0])
                    mean_acc.append(score)
                    pbar.update(1)

                acc = np.array(mean_acc).mean()
                ens_acc.append({"Genre": genre, "Accuracy": acc})

        ens_acc = pd.DataFrame(ens_acc)
        return self.process_accuracies(ens_acc)[0]
        # ens_acc = ens_acc.append({"Genre": "Average", "Accuracy": ens_acc["Accuracy"].mean()}, ignore_index=True)
        #
        # if kwargs.get("add_to_acc", None) is not None:
        #     if isinstance(kwargs["add_to_acc"], list):
        #         ens_acc.insert(0, "Model Name", ens_str)
        #         kwargs["add_to_acc"].append(ens_acc)
        #     if isinstance(kwargs["add_to_acc"], dict):
        #         kwargs["add_to_acc"].update({ens_str: ens_acc})
        # if kwargs.get("disp_acc", False):
        #     display_df(ens_acc, f"<h4>{ens_str} Accuracy by Genre</h4>")
        # return ens_acc

    def load_pre_calculated(self, load: bool, file_name: str):
        if load:
            if os.path.exists(str(self.data_paths["all"].joinpath(file_name))):
                loaded = pickle.load(open(str(self.data_paths["all"].joinpath(file_name)), "rb+"))
                return loaded
        return None

    def print_num_features(self):
        for genre, data in self.model_loader.data.groupby(by="@Genre", as_index=False):
            if genre == "Horror":
                continue
            if "gram" in self.model.lower():
                temp = self.tfi_ngram(data.drop(columns=["Book #", "@Genre", "@Outcome"]))
                print(genre, filter_out_zeros(temp).shape)
            else:
                temp = data
                print(genre, filter_out_zeros(temp.drop(columns=["Book #", "@Genre", "@Outcome"])).shape)
            gc.collect()

    def score_books_by_genre(self, weights: Dict, data_: Optional[pd.DataFrame] = None):
        if data_ is None:
            df = self.model_loader.data.copy()
        else:
            df = data_.copy()

        df["@Score"] = 0
        scores = []
        groupby = df.groupby(by="@Genre", as_index=False)

        if "@Downloads" not in df.columns:
            df = self.merge_with_downloads(df)

        with self.tqdm(total=len(groupby.groups)) as pbar:
            for genre, data in df.groupby(by="@Genre", as_index=False):
                if genre == "Horror":
                    continue
                pbar.set_postfix_str(f"-- SCORING -- {genre}")
                score = data[["Book #", "@Genre"] + list(weights[genre]["Feature"]) + ["@Outcome", "@Downloads", "@Score"]].copy()
                score["@Score"] = score.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads", "@Score"]).mul(weights[genre].set_index("Feature")["Weight"], axis=1).sum(axis=1)
                scores.append(score)
                pbar.update(1)

        df = pd.concat(scores).fillna(0)
        cols = list(df.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads", "@Score"]).columns)
        cols.sort()
        df = df[["Book #", "@Genre"] + cols + ["@Outcome", "@Downloads", "@Score"]]

        return df
