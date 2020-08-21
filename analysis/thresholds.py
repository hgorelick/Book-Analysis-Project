import pickle
from typing import Optional, List, Tuple, Dict, Union

import pandas as pd
from matplotlib import pyplot as plt

from loading_utils.models import ModelLoader
from notebook_utils.constants import NO_HORROR, scaler
from notebook_utils.predictions import Predictor
from notebook_utils.utils import display_df, setup_markers, finish_plot, setup_search_plot, filter_out_zeros


class Threshold:
    def __init__(self, high: pd.DataFrame, low: pd.DataFrame, ht: float, lt: float, ht_d: int, lt_d: int):
        self.high: pd.DataFrame = high
        self.low: pd.DataFrame = low
        self.hl: int = len(high)
        self.ll: int = len(low)
        self.diff: int = abs(self.hl - self.ll)
        self.ht: float = ht
        self.lt: float = lt
        self.ht_d: int = ht_d
        self.lt_d: int = lt_d
        self.classifications = self.get_classifications()

    def get_classifications(self):
        self.high["@Outcome"] = "SUCCESSFUL"
        self.low["@Outcome"] = "FAILURE"
        return pd.concat([self.high, self.low])  # .reset_index(drop=True)

    def __repr__(self):
        return f"hl: {self.hl}, ll: {self.ll}, ht: {self.ht}, lt: {self.lt}, ht_d: {self.ht_d}, lt_d: {self.lt_d}"

    def __str__(self):
        return f"hl: {self.hl}, ll: {self.ll}, ht: {self.ht}, lt: {self.lt}, ht_d: {self.ht_d}, lt_d: {self.lt_d}"

    def __eq__(self, other):
        return self.hl == other.hl and self.ll == other.ll and self.ht == other.ht \
               and self.lt == other.lt and self.ht_d == other.ht_d and self.lt_d == other.lt_d

    def __ne__(self, other):
        return self.hl != other.hl or self.ll != other.ll or self.ht != other.ht \
               or self.lt != other.lt or self.ht_d != other.ht_d or self.lt_d != other.lt_d

    def __hash__(self):
        return hash((self.hl, self.ll, self.ht, self.lt, self.ht_d, self.lt_d))


# TODO: Plot threshold vs accuracy
def finalize(thresh_data, indexes, best_df) -> pd.DataFrame:
    thresh_df = pd.DataFrame(thresh_data, index=indexes)
    merged = pd.merge(best_df.reset_index(), thresh_df.reset_index(), how="outer", on=["Genre", "index"]).drop(columns=["index"])
    results = []
    for genre, data in merged.groupby(by="Genre"):
        results.append(data.loc[data["High Length"] == data["High Length"].max()])
    results_df = pd.concat(results)
    return results_df


class ThresholdApplier(Predictor):

    def __init__(self, model: str = "", first_1k: bool = False, by_genre: bool = False, jupyter: bool = False,
                 model_loader: Optional[ModelLoader] = None, estimator: str = "svm", scale: bool = True,
                 load: bool = True, bins: bool = False, **kwargs):
        super().__init__(model, first_1k, by_genre, jupyter, model_loader, estimator, scale=scale, **kwargs)
        self.scale = scale
        self.load = load
        self.bins = bins
        self.file_name = f"{self.model} {'bis' if self.bins else 'thresholds'}{' by genre' if self.by_genre else ''}"
        if not self.scale:
            self.file_name += " not scaled"

    def apply_thresholds_by_genre(self, idx: Dict, accs: Dict, weights: Dict, preds: Dict, thresholds: Dict) -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]:
        if "@Downloads_x" in self.model_loader.data.columns:
            self.model_loader.data.drop(columns=["@Downloads_x"], inplace=True)
        if "@Downloads_y" in self.model_loader.data.columns:
            self.model_loader.data.drop(columns=["@Downloads_y"], inplace=True)

        max_accs = []
        best_data = []

        for genre, data in self.model_loader.data.groupby(by="@Genre", as_index=False):
            if genre == "Horror":
                continue
            modded = []
            for i in range(len(accs[genre])):
                mod = accs[genre][i].loc[accs[genre][i]["Genre"] != "Average"]
                modded.append(mod)

            modded_df = pd.concat(modded).reset_index(drop=True)
            max_accs.append(modded_df.loc[modded_df["Accuracy"] == modded_df["Accuracy"].max()])

            temp = data.drop(columns=["@Outcome"])  # .reset_index(drop=True)
            best_merged = self.add_outcome(temp, thresholds[genre][idx[genre]])
            best_data.append(best_merged)

        best_data_df = pd.concat(best_data)  # .reset_index(drop=True).sort_values(by=["@Genre"])
        self.model_loader.data = best_data_df

        best_df = pd.concat(max_accs)

        thresh_data = []
        indexes = []
        for index, data in best_df.iterrows():
            genre = data["Genre"]
            thresh_data.append({"Genre": genre, "Low Thresh": thresholds[genre][index].lt_d, "High Thresh": thresholds[genre][index].ht_d,
                                "Low Length": thresholds[genre][index].ll, "High Length": thresholds[genre][index].hl})
            indexes.append(index)

        best_accs, best_weights, best_preds, results = self.process_threshold_results_by_genre(accs, weights, preds, thresh_data, best_df, indexes)

        results.sort_values(by=["Genre"], inplace=True)
        results = results.append({"Genre": "Average", "Accuracy": results["Accuracy"].mean(),
                                  "Low Thresh": results["Low Thresh"].mean(), "High Thresh": results["High Thresh"].mean(),
                                  "Low Length": results["Low Length"].mean(), "High Length": results["High Length"].mean()},
                                 ignore_index=True)

        if self.jupyter:
            display_df(results)

        return best_accs, best_weights, best_preds, results

    def test_thresholds_by_genre(self, add_to_acc: Optional[List] = None) -> Tuple[Dict[str, int], Dict, Dict, Dict, Dict[str, List[Threshold]]]:
        """

        :param add_to_acc:
        :return:
        """
        thresholds = self.get_download_thresholds_by_genre()
        # print(thresholds)
        accuracies = {genre: [] for genre in NO_HORROR}
        weights = {genre: [] for genre in NO_HORROR}
        preds = {genre: [] for genre in NO_HORROR}

        best_indexes = {genre: 0 for genre in NO_HORROR}
        bar_length = sum(len(thresholds[genre]) for genre in NO_HORROR)
        with self.tqdm(total=bar_length) as pbar:
            for genre in NO_HORROR:
                best = 0.0
                pbar.set_postfix_str(f"-- TESTING THRESHOLDS - {genre}")
                data = self.model_loader.data.loc[self.model_loader.data["@Genre"] == genre].copy()
                data = filter_out_zeros(data.drop(columns=["@Outcome"]))  # .reset_index(drop=True)
                for i in range(len(thresholds[genre])):
                    merged = self.add_outcome(data, thresholds[genre][i])
                    acc, ws, p = self.predict_by_genre([genre], add_to_acc=add_to_acc, disp_acc=False, disp_weights=False,
                                                       show_pbar=False, temp=merged)
                    if acc["Accuracy"].values[0] > best:
                        best = acc["Accuracy"].values[0]
                        best_indexes[genre] = i
                    accuracies[genre].append(acc)
                    weights[genre].append(ws[genre])
                    preds[genre].append(p[genre])
                    pbar.update(1)

        return best_indexes, accuracies, weights, preds, thresholds

    def get_download_thresholds_by_genre(self):
        loaded = self.load_pre_calculated(self.load, self.file_name)
        if loaded is not None:
            return loaded

        with_downloads = self.merge_with_downloads()
        groupby_genre = with_downloads.groupby(by="@Genre")
        thresholds = {genre: [] for genre in NO_HORROR}

        with self.tqdm(total=len(groupby_genre)) as pbar:
            for genre, data in groupby_genre:
                pbar.set_postfix_str(f"-- FINDING THRESHOLDS -- {genre}")
                group = data.copy().sort_values(by=["@Outcome"], ascending=False)
                # group.reset_index(drop=True, inplace=True)
                thresholds[genre] = self.widen_threshold(group)
                pbar.update(1)

        with open(str(self.data_paths["all"].joinpath(self.file_name)), "wb+") as f:
            pickle.dump(thresholds, f)

        return thresholds

    @staticmethod
    def process_threshold_results_by_genre(accuracies: Dict, weights: Dict, preds: Dict,
                                           thresh_data: List, best_df: pd.DataFrame, indexes: List):
        best_accs = pd.DataFrame([accuracies[genre][idx].iloc[0, :] for genre, idx in zip(NO_HORROR, indexes)]).reset_index(drop=True)
        best_weights = {genre: weights[genre][idx] for genre, idx in zip(NO_HORROR, indexes)}
        best_preds = {genre: preds[genre][idx] for genre, idx in zip(NO_HORROR, indexes)}
        return best_accs, best_weights, best_preds, finalize(thresh_data, indexes, best_df)

    def apply_thresholds(self, accs: List, weights: List, preds: List, thresholds: List[Threshold]) -> Tuple[List, List, List, pd.DataFrame]:
        # modded = []
        accs_df = pd.concat(accs).reset_index(drop=True)
        thresh_df = pd.DataFrame([{"LL": t.ll, "HL": t.hl, "Diff": t.diff, "LT": t.lt_d, "HT": t.ht_d} for t in thresholds])
        results_df = pd.concat([accs_df, thresh_df], axis=1)

        best = results_df.loc[results_df["Diff"] < 500]
        best = best.loc[best["Accuracy"] == best["Accuracy"].max()]

        idx = best.index.values[0]

        best_accs = accs[idx]
        best_weights = weights[idx]
        best_preds = preds[idx]

        data = self.model_loader.data.copy().drop(columns=["@Outcome"])  # .reset_index(drop=True)
        best_merged = self.add_outcome(data, thresholds[idx])
        self.model_loader.data = best_merged

        if self.jupyter:
            display_df(results_df)

        return best_accs, best_weights, best_preds, results_df

    def test_thresholds(self, add_to_acc: Optional[List] = None) -> Tuple[List, List, List, List]:
        thresholds = self.get_download_thresholds()
        accuracies = []
        weights = []
        preds = []

        best = 0.0
        # best_idx = 0
        with self.tqdm(total=len(thresholds) * 5, postfix="-- TESTING THRESHOLDS") as pbar:
            for i in range(len(thresholds)):
                data = filter_out_zeros(self.model_loader.data.copy().drop(columns=["@Outcome"]))  # .reset_index(drop=True)
                merged = self.add_outcome(data, thresholds[i])
                acc, ws, p = self.predict_all(temp=merged, add_to_acc=add_to_acc, pbar=pbar, disp_acc=False, disp_weights=False)
                if acc["Accuracy"].values[0] > best:
                    best = acc["Accuracy"].values[0]
                    # best_idx = i
                accuracies.append(acc)
                weights.append(ws)
                preds.append(p)

        return accuracies, weights, preds, thresholds

    def get_download_thresholds(self):
        loaded = self.load_pre_calculated(self.load, self.file_name)
        if loaded is not None:
            return loaded

        with_downloads = self.merge_with_downloads()

        print("Finding thresholds for all books...")
        thresholds = self.widen_threshold(with_downloads)
        print(f"{len(thresholds)} found!")

        with open(str(self.data_paths["all"].joinpath(self.file_name)), "wb+") as f:
            pickle.dump(thresholds, f)

        return thresholds

    @staticmethod
    def add_outcome(data: pd.DataFrame, threshold: Threshold):
        merged = pd.merge(threshold.classifications, data, on=["Book #"])  # .drop(columns=["Downloads"])
        special_cols = ["Book #", "@Genre", "@Outcome"]
        od = ["@Outcome"]
        if "@Downloads" in merged.columns:
            special_cols += ["@Downloads"]
            od += ["@Downloads"]
        not_special = [col for col in merged.columns if col not in special_cols]
        merged = merged[["Book #", "@Genre"] + not_special + od]
        return merged

    def widen_threshold(self, data: pd.DataFrame):
        if self.scale:
            data["@Outcome"] = scaler.fit_transform(data[["@Outcome"]])
        median = data["@Outcome"].median()
        hidx = 1
        lidx = 0
        high, low, threshold = self.get_high_low(data, median, median)
        threshold_data = [threshold]
        ht = data.loc[data["@Outcome"] > median].iloc[-hidx]["@Outcome"]
        lt = data.loc[data["@Outcome"] < median].iloc[lidx]["@Outcome"]
        while len(high) >= 100 and len(low) >= 100:
            high, low, threshold = self.get_high_low(data, ht, lt)
            if threshold not in threshold_data:
                if not self.by_genre and threshold.diff < 500:
                    threshold_data.append(threshold)
                else:
                    threshold_data.append(threshold)
            hidx += 1
            lidx += 1
            ht = data.loc[data["@Outcome"] > median].iloc[-hidx]["@Outcome"]
            lt = data.loc[data["@Outcome"] < median].iloc[lidx]["@Outcome"]

        thresholds = list(set([t for t in threshold_data if t.diff <= 1]))
        if len(thresholds) == 0:
            thresholds = list(set(threshold_data))

        return thresholds

    @staticmethod
    def get_high_low(data: pd.DataFrame, ht: Union[int, float], lt: Union[int, float]):
        high = data.loc[data["@Outcome"] > ht][["Book #", "@Outcome", "@Downloads"]].copy()
        low = data.loc[data["@Outcome"] < lt][["Book #", "@Outcome", "@Downloads"]].copy()

        ht_d = high["@Downloads"].min()
        lt_d = low["@Downloads"].max()

        threshold = Threshold(high, low, ht, lt, ht_d, lt_d)

        return high, low, threshold

    @staticmethod
    def process_threshold_results(accuracies: List, weights: List, preds: List, thresh_data: List, best_df: pd.DataFrame, indexes: List):
        best_accs = [accuracies[idx] for idx in indexes]
        best_weights = [weights[idx] for idx in indexes]
        best_preds = [preds[idx] for idx in indexes]
        return best_accs, best_weights, best_preds, finalize(thresh_data, indexes, best_df)

    def plot_thresholds_by_genre(self, thresh_df_: pd.DataFrame, marker_size: int = 10, genre_list: List = NO_HORROR,
                                 colors: Optional[Dict] = None, for_paper: bool = False):

        width = 15 if for_paper else 30
        height = 7 if for_paper else 17
        fig, axes, thresh_df, max_steps = setup_search_plot(thresh_df_, "Margin Width", width, height)
        tuned_params = []
        markers = setup_markers(genre_list)
        line_width = marker_edge_width = 1 if for_paper else 2

        # if use_markers:
        for (genre, data), m in zip(thresh_df.groupby(by="Genre"), markers):
            self.process_thresh_df(data, genre, tuned_params)
            if colors is None:
                data[["Margin Width", "Accuracy"]].plot(x="Margin Width", ax=axes, rot=45,
                                                        marker=m if len(genre_list) <= len(NO_HORROR) else "",
                                                        markersize=marker_size, markeredgewidth=marker_edge_width,
                                                        fillstyle="none", linewidth=line_width)
            else:
                data[["Margin Width", "Accuracy"]].plot(x="Margin Width", ax=axes, rot=45, color=colors[genre],
                                                        marker=m if len(genre_list) <= len(NO_HORROR) else "",
                                                        markersize=marker_size, markeredgewidth=marker_edge_width,
                                                        fillstyle="none", linewidth=line_width)
        # else:
        #     for genre in genre_list:
        #         self.process_thresh_df(thresh_df, tuned_params)
        #         if colors is None:
        #             thresh_df[thresh_df["Genre"] == genre][["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=0, linewidth=2)
        #         else:
        #             thresh_df[thresh_df["Genre"] == genre][["Number of Features", "Accuracy"]].plot(x="Number of Features", ax=axes, rot=0, color=colors[genre], linewidth=2)

        tuned_params_df = finish_plot(axes, thresh_df, max_steps, tuned_params, "Margin Width", for_paper=for_paper)

        axes.legend(genre_list, bbox_to_anchor=(0.075, 1.23), loc="upper left", fontsize=16, ncol=len(genre_list) // 3)
        plt.margins(x=0.01, y=0.05)
        plt.show()

        return tuned_params_df

    def plot_thresholds(self, thresh_df_: pd.DataFrame, marker_size: int = 10, for_paper: bool = False):

        width = 15 if for_paper else 30
        height = 8 if for_paper else 17
        fig, axes, thresh_df, max_steps = setup_search_plot(thresh_df_, "Margin Width", width, height)
        tuned_params = []
        line_width = marker_edge_width = 1 if for_paper else 2

        thresh_df[["Margin Width", "Accuracy"]].plot(x="Margin Width", ax=axes, rot=0, marker="o", markersize=marker_size,
                                                     markeredgewidth=marker_edge_width, fillstyle="none", linewidth=line_width)

        tuned_params_df = finish_plot(axes, thresh_df, max_steps, tuned_params, "Margin Width", True, for_paper=for_paper)

        axes.legend(labels=["All Books"], loc="upper right", fontsize=12)
        plt.margins(x=0.01, y=0.05)
        plt.show()

        return tuned_params_df

    @staticmethod
    def process_thresh_df(df: pd.DataFrame, genre: str, tuned_params: List):
        best = df["Accuracy"].max()
        best_width = df.loc[df["Accuracy"] == best, "Margin Width"].values[0]
        tuned_params.append({"Genre": genre, "Accuracy": best, "Margin Width": best_width})
