from typing import Optional

import pandas as pd
import warnings
from sklearn.preprocessing import KBinsDiscretizer

from loading_utils.models import ModelLoader
from notebook_utils.constants import NO_HORROR
from notebook_utils.predictions import Predictor


class BinApplier(Predictor):

    def __init__(self, model: str = "", first_1k: bool = False, by_genre: bool = False, jupyter: bool = False,
                 model_loader: Optional[ModelLoader] = None, scale: bool = True, load: bool = True, **kwargs):
        super().__init__(model, first_1k, by_genre, jupyter, model_loader, estimator="lr", scale=scale, **kwargs)
        self.scale = scale
        self.load = load
        self.file_name = f"{self.model} bins{' by genre' if self.by_genre else ''}"
        if not self.scale:
            self.file_name += " not scaled"

    def apply_bins(self):
        if self.by_genre:
            return self._test_bins_by_genre()
        else:
            return self._test_bins(self.model_loader.data, "All")

    def _test_bins_by_genre(self):
        # loaded = self.load_pre_calculated(self.load, self.file_name)
        # if loaded is not None:
        #     return loaded
        groupby_genre = self.model_loader.data.groupby(by="@Genre")
        bins = {genre: {} for genre in NO_HORROR}

        with self.tqdm(total=len(groupby_genre)) as pbar:
            for genre, data in groupby_genre:
                pbar.set_postfix_str(f"-- FINDING BINS -- {genre}")
                group = data.copy().sort_values(by=["@Outcome"], ascending=False)
                bins[genre] = self._test_bins(group, genre)
                pbar.update(1)

        # with open(str(self.data_paths["all"].joinpath(self.file_name)), "wb+") as f:
        #     pickle.dump(bins, f)

        return bins

    def _test_bins(self, data: pd.DataFrame, genre: str):
        # if self.scale and self.by_genre:
        #     data["@Outcome"] = scaler.fit_transform(data[["@Outcome"]])

        best_acc = 0.0
        bin_results = {}
        best_binned = pd.DataFrame()
        best_weights = pd.DataFrame()
        best_preds = pd.DataFrame()

        k = len(data)
        best_k = k
        i = 0
        with self.tqdm(total=len(data) * 5) as pbar:
            while True:
                pbar.set_postfix_str(f"-- TESTING {k} BINS -- {genre}")
                if best_acc == 1.0:
                    print(f"{genre} predicted downloads bins exhausted at {k} bins (100%)")
                    pbar.update((len(data) * 5) - i)
                    break

                bins = data.copy()
                kb = KBinsDiscretizer(n_bins=k, encode="ordinal")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bins["@Outcome"] = kb.fit_transform(bins[["@Outcome"]])

                acc, w, p = self.predict_all(temp=bins, pbar=pbar, disp_acc=False, disp_weights=False)
                bin_results[k] = {"Genre": genre, "Bins": k, "Accuracy": acc["Accuracy"].values[0]}

                if acc["Accuracy"].values[0] > best_acc:
                    best_acc = acc["Accuracy"].values[0]
                    best_binned = bins
                    best_weights = w
                    best_preds = p
                    best_k = k
                    k += k // 2
                elif bin_results[best_k]["Accuracy"] > acc["Accuracy"].values[0]:
                    if k - (best_k // 4) == best_k:
                        k -= k // 2
                    else:
                        k -= best_k // 4
                else:
                    k -= k // 2

                if k < 5:
                    print(f"{genre} predicted downloads bins exhausted at 5 bins")
                    pbar.update((len(data) * 5) - i)
                    break

                elif k > len(data):
                    print(f"{genre} predicted downloads bins exhausted at {k - (k // 2)} bins")
                    pbar.update((len(data) * 5) - i)
                    break

                pbar.update(1)
                i += 1

        bin_results_df = pd.DataFrame(list(bin_results.values())).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
        return {"Max Accuracy": best_acc, "Best Binned": best_binned,
                "Best Weights": best_weights, "Best Preds": best_preds,
                "Full Results": bin_results_df}
