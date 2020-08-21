__all__ = ['FeatureReducer']

from math import ceil
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from IPython.display import display, HTML
from matplotlib import pyplot as plt

from notebook_utils.constants import NO_HORROR
from notebook_utils.predictions import Predictor
from notebook_utils.utils import setup_markers, finish_plot, setup_search_plot


class FeatureReducer(Predictor):

    # def __init__(self, model: str = "", first_1k: bool = False, by_genre: bool = False, jupyter: bool = False,
    #              model_loader: Optional[ModelLoader] = None, **kwargs):
    #     super(FeatureReducer, self).__init__(model, first_1k, by_genre, jupyter, model_loader, **kwargs)

    def get_df_for_reduction(self, g: Optional[str] = None, g2: Optional[str] = None):
        if g is None:
            if "gram" in self.model:
                out_col = self.model_loader.data[["Book #", "@Genre", "@Outcome", "@Downloads"]].copy().reset_index(drop=True)
                data = self.tfi_ngram(self.model_loader.data)
                data.insert(0, "@Genre", out_col["@Genre"])
                data.insert(0, "Book #", out_col["Book #"])
                data["@Outcome"] = out_col["@Outcome"]
                data["@Downloads"] = out_col["@Downloads"]
                return data
            else:
                return self.model_loader.data.copy()

        if g2 is None:
            if "gram" in self.model:
                out_col = self.model_loader.data[self.model_loader.data["@Genre"] == g][["Book #", "@Outcome"]].copy().reset_index(drop=True)
                data = self.tfi_ngram(self.model_loader.data[self.model_loader.data["@Genre"] == g])
                data.insert(0, "@Genre", g)
                data.insert(0, "Book #", out_col["Book #"])
                data["@Outcome"] = out_col["@Outcome"]
                return data
            else:
                return self.model_loader.data[self.model_loader.data["@Genre"] == g].copy()

        else:
            if "gram" in self.model:
                out_col = self.model_loader.data[(self.model_loader.data["@Genre"] == g) |
                                                 (self.model_loader.data["@Genre"] == g2)][["Book #", "@Genre", "@Outcome"]]\
                                                .copy()\
                                                .reset_index(drop=True)
                data = self.tfi_ngram(self.model_loader.data[(self.model_loader.data["@Genre"] == g) | (self.model_loader.data["@Genre"] == g2)])
                data.insert(0, "@Genre", out_col["@Genre"])
                data.insert(0, "Book #", out_col["Book #"])
                data["@Outcome"] = out_col["@Outcome"]
                return data
            else:
                return self.model_loader.data[(self.model_loader.data["@Genre"] == g) | (self.model_loader.data["@Genre"] == g2)].copy()

    def reduce_features_by_genre(self, model_weights: Dict, og_acc: pd.DataFrame, og_preds: Dict, data: Optional[pd.DataFrame] = None,
                                 genre_list: List = NO_HORROR, g_predict: Optional[str] = None, **kwargs):
        header = f"<h4>Performing exhaustive parameter search for feature reduction on {self.model} by genre"
        header += f"for {g_predict} Genre Prediction</h4>" if g_predict is not None else "</h4>"
        display(HTML(header))

        exhausted = []
        reduced_features = {}
        reduced_preds = {}
        for genre, data in self.model_loader.data.groupby(by="@Genre", as_index=False):
            if genre == "Horror":
                continue
            results = self.reduce_features(model_weights[genre], og_acc.loc[og_acc["Genre"] == genre], og_preds[genre], data, genre=genre, **kwargs)
            exhausted.append(results[0])
            reduced_features[genre] = results[1]
            reduced_preds[genre] = results[2]

        exhausted_df = pd.concat(exhausted)
        return exhausted_df, reduced_features, reduced_preds

    # def _reduce_features_by_genre(self, model_weights: pd.DataFrame, genre: str, g_predict: Optional[str] = None,
    #                               og_acc: Optional[pd.DataFrame] = None):
    #
    #     og_copy = og_acc.copy()
    #     og_copy = og_copy[og_copy["Genre"] != "Average"]
    #
    #     og_copy.insert(1, "Step", -0.25)
    #     n_feats = [len(model_weights[k]) for k, v in model_weights.items()]
    #     og_copy["Number of Features"] = n_feats
    #
    #     exhaustive = og_copy.to_dict("records")
    #     reduced_features = {genre: model_weights[genre].copy() for genre in genre}
    #
    #     steps = np.arange(0, 100 + 0.25, 0.25)
    #     with self.tqdm(total=len(steps)) as pbar:
    #         pbar.set_postfix_str(f" -- {genre}")
    #         best_acc = og_acc["Accuracy"].mean()
    #
    #         i = 0
    #         step = 0
    #         while True:
    #             avg_weight = model_weights["Weight"].mean()
    #             std_dev = model_weights["Weight"].std()
    #             threshold = avg_weight + (step * std_dev)
    #             param_results = model_weights[model_weights["Weight"].abs() >= threshold]
    #
    #             if len(param_results) < 5:
    #                 print(f"{genre} exhausted at {step} deviations above the mean")
    #                 pbar.update(len(steps) - i)
    #                 break
    #
    #             if best_acc == 1.0:
    #                 print(f"{genre} best param at {step - 0.25} deviations above the mean")
    #                 pbar.update(len(steps) - i)
    #                 break
    #
    #             elif len(param_results) == len(model_weights["Weight"]):
    #                 i += 1
    #                 step += 0.25
    #                 pbar.update(1)
    #                 continue
    #
    #             param_results_df = self.filter_out_reduction(param_results, genre)
    #
    #             if g_predict is not None:
    #                 param_acc, param_weights = self.predict_genre(how=g_predict, genre_list=[genre], searching=True,
    #                                                               disp_acc=False, disp_weights=False,
    #                                                               show_pbar=False)
    #                 if g_predict == "one_v_one":
    #                     step_acc = param_acc.loc[param_acc["Genre"] == (genre[0], genre[1]), "Accuracy"].values[0]
    #                 else:
    #                     step_acc = param_acc.loc[param_acc["Genre"] != "Average", "Accuracy"].values[0]
    #             else:
    #                 param_acc, param_weights, _ = self.predict_all(temp=param_results_df, searching=True, disp_acc=False, disp_weights=False,
    #                                                             show_pbar=False)
    #                 step_acc = param_acc.loc[param_acc["Genre"] != "Average", "Accuracy"].values[0]
    #
    #             if step_acc > best_acc:
    #                 best_acc = step_acc
    #                 reduced_features[genre] = param_weights[genre].copy()
    #
    #             exhaustive.append({"Genre": genre, "Step": step, "Accuracy": step_acc, "Number of Features": len(param_weights[genre])})
    #
    #             i += 1
    #             step += 0.25
    #             pbar.update(1)
    #
    #     exhaustive_df = pd.DataFrame(exhaustive)
    #     return exhaustive_df, reduced_features

    def reduce_features(self, model_weights: pd.DataFrame, og_acc: Optional[pd.DataFrame] = None, og_preds: Optional[pd.Series] = None,
                        data: Optional[pd.DataFrame] = None, genre: str = "", **kwargs):
        if genre == "":
            display(HTML(f"<h4>Performing exhaustive parameter search for feature reduction on"
                         f" {self.model} {'by' if self.by_genre else 'across'} genres</h4>"))

        og_copy = og_acc.copy()
        og_copy = og_copy[og_copy["Genre"] != "Average"]

        og_copy.insert(1, "Step", -0.25)
        n_feats = len(model_weights)
        og_copy["Number of Features"] = [n_feats]

        exhaustive = og_copy.to_dict("records")
        reduced_features = model_weights.copy()
        reduced_preds = og_preds.copy()

        steps = np.arange(0, 100.25, 0.25)
        with self.tqdm(total=len(steps)) as pbar:
            if genre != "":
                pbar.set_postfix_str(f"-- {genre}")
            best_acc = og_acc["Accuracy"].mean()

            i = 0
            step = 0
            decrease_count = 0
            while True:
                if decrease_count > 10:
                    print(f"{self.model if genre == '' else genre} exhausted at {step} deviations,"
                          f" {i} steps above the mean (found best)")
                    pbar.update(len(steps) - i)
                    break

                if best_acc == 1.0:
                    print(f"{self.model if genre == '' else genre} exhausted at {step} deviations,"
                          f" {i} steps above the mean (100% accuracy achieved)")
                    pbar.update(len(steps) - i)
                    break

                avg_weight = model_weights["Weight"].mean()
                std_dev = model_weights["Weight"].std()
                threshold = avg_weight + (step * std_dev)
                param_results = model_weights[model_weights["Weight"].abs() >= threshold]

                if len(param_results) < (n_feats * 0.01):
                    print(f"{self.model if genre == '' else genre} exhausted at {step} deviations,"
                          f" {i} steps above the mean (1% reduction limit reached)")
                    pbar.update(len(steps) - i)
                    break

                elif len(param_results) == len(model_weights["Weight"]):
                    i += 1
                    step += 0.25
                    pbar.update(1)
                    continue

                param_results_df = self.filter_out_reduction(param_results, genre, df=data)
                # print(param_results_df)
                # display_df(get_display_df(param_results_df))
                param_acc, param_weights, preds = self.predict_all(temp=param_results_df, disp_acc=False, disp_weights=False, show_pbar=False,
                                                                   reducing=True, **kwargs)

                step_acc = param_acc.loc[param_acc["Genre"] != "Average", "Accuracy"].values[0]

                if step_acc > best_acc:
                    best_acc = step_acc
                    reduced_features = param_weights.copy()
                    reduced_preds = preds.copy()
                    decrease_count = 0
                else:
                    decrease_count += 1

                exhaustive.append({"Genre": genre if genre != "" else "All", "Step": step,
                                   "Accuracy": step_acc, "Number of Features": len(param_weights)})

                i += 1
                step += 0.25
                pbar.update(1)

        exhaustive_df = pd.DataFrame(exhaustive)
        return exhaustive_df, reduced_features, reduced_preds

    def filter_out_reduction(self, param_results: pd.DataFrame, genre: str = "", df: Optional[pd.DataFrame] = None):
        col_filter = set(param_results["Feature"])
        cols = ["Book #", "@Genre"] + list(col_filter) + ["@Outcome", "@Downloads"]
        if df is not None:
            data = df.copy()
        else:
            data = self.model_loader.data

        if "gram" in self.model.lower():
            data = self.get_df_by_name(data)[cols]
            return data

        if genre != "":
            param_results_df = data.loc[data["@Genre"] == genre]
        else:
            param_results_df = data
        param_results_df = param_results_df[cols]  # .reset_index(drop=True)
        return param_results_df

    def plot_exhausted_by_genre(self, exh_df_: pd.DataFrame, marker_size: int = 10, genre_list: List = NO_HORROR, colors: Optional[Dict] = None,
                                use_markers: bool = True, for_paper: bool = False):

        width = 15 if for_paper else 30
        height = 7 if for_paper else 17
        fig, axes, exh_df, max_steps = setup_search_plot(exh_df_, "Number of Features", width, height)
        tuned_params = []
        tuned_params_dict = {}
        markers = setup_markers(genre_list)

        line_width = marker_edge_width = 1 if for_paper else 2

        if use_markers:
            for (genre, data), m in zip(exh_df.groupby(by="Genre", as_index=False), markers):
                self.process_exh_df(exh_df, genre, tuned_params, tuned_params_dict)
                if colors is None:
                    data[["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=45,
                                                                  marker=m if len(genre_list) <= len(NO_HORROR) else "",
                                                                  markersize=marker_size,
                                                                  markeredgewidth=marker_edge_width if genre != "All" else 2,
                                                                  fillstyle="none", linewidth=line_width if genre != "All" else 2)
                else:
                    data[["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=45, color=colors[genre],
                                                                  marker=m if len(genre_list) <= len(NO_HORROR) else "",
                                                                  markersize=marker_size,
                                                                  markeredgewidth=marker_edge_width if genre != "All" else 2,
                                                                  fillstyle="none", linewidth=line_width if genre != "All" else 2)
        else:
            for genre, data in exh_df.groupby(by="Genre", as_index=False):
                self.process_exh_df(exh_df, genre, tuned_params, tuned_params_dict)
                if colors is None:
                    data[["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=0, linewidth=2)
                else:
                    data[["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=0, color=colors[genre], linewidth=2)

        tuned_params_df = finish_plot(axes, exh_df, max_steps, tuned_params, "Number of Features", for_paper=for_paper, reverse=True)

        if for_paper:
            labels = genre_list + ["All Books"]
            labels.sort()
            axes.legend(labels=labels, bbox_to_anchor=(0.07, 1.23), loc="upper left", fontsize=16, ncol=ceil(len(labels) / 4))
        else:
            axes.legend(genre_list, bbox_to_anchor=(0, 1.05), loc="upper left", fontsize=14, ncol=len(genre_list))
        plt.margins(x=0.01, y=0.05)
        plt.show()

        return tuned_params_df

    def plot_exhausted(self, exh_df_: pd.DataFrame, marker_size: int = 10, for_paper: bool = False):
        width = 15 if for_paper else 30
        height = 10 if for_paper else 17
        fig, axes, exh_df, max_steps = setup_search_plot(exh_df_, "Number of Features", width, height)
        tuned_params = []
        line_width = marker_edge_width = 1 if for_paper else 2

        best = exh_df["Accuracy"].max()
        best_param = exh_df.loc[(exh_df["Accuracy"] == best), "Step"].values[0]
        n_features = exh_df.loc[(exh_df["Accuracy"] == best), "Number of Features"].values[0]
        tuned_params.append({"Genre": "All", "Deviations": best_param, "Accuracy": best, "Number of Features": n_features})
        exh_df[["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=45, marker="o", markersize=marker_size,
                                                        markeredgewidth=marker_edge_width, fillstyle="none", linewidth=line_width)

        tuned_params_df = finish_plot(axes, exh_df, max_steps, tuned_params, "Number of Features", for_paper=for_paper, reverse=True)

        axes.legend(labels=["All Books"], bbox_to_anchor=(0, 1), loc="upper left", fontsize=24)

        plt.margins(x=0.01, y=0.05)
        plt.show()
        return tuned_params_df

    @staticmethod
    def process_exh_df(exh_df: pd.DataFrame, genre: str, tuned_params: List, tuned_params_dict: Dict):
        best = exh_df[(exh_df["Genre"] == genre)]["Accuracy"].max()
        best_param = exh_df.loc[(exh_df["Genre"] == genre) & (exh_df["Accuracy"] == best), "Step"].values[0]
        n_features = exh_df.loc[(exh_df["Genre"] == genre) & (exh_df["Accuracy"] == best), "Number of Features"].values[0]
        tuned_params.append({"Genre": genre, "Deviations": best_param, "Accuracy": best, "Number of Features": n_features})
        tuned_params_dict[genre] = {"Deviations": best_param, "Accuracy": best}
