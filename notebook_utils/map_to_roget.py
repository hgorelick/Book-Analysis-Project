__all__ = ['invert_roget', 'RogetMapper']

from collections import defaultdict
from typing import Optional, List, Dict, Union

import pandas as pd
from IPython.display import display, HTML

from loading_utils.models import ModelLoader
from notebook_utils.constants import *
from notebook_utils.feature_reduction import FeatureReducer
from notebook_utils.utils import display_df, process_and_scale, filter_out_zeros


def invert_roget():
    roget_invert = []

    for idx, row in roget_thesaurus.roget_df.iterrows():
        for word in set(row["Words"]):
            roget_invert.append({"Word": word, "Category": row["Category"], "Level3": row["Level3"], "Level2": row["Level2"],
                                 "Level1": row["Level1"], "Section": row["Section"], "Class": row["Class"]})

    roget_thesaurus_df = pd.DataFrame(roget_invert)
    return roget_thesaurus_df


# def make_clean_model_df(model_data: List):
#     clean_df = pd.DataFrame(model_data).fillna(0).rename(columns={"_genre": "@Genre", "_outcome": "@Outcome"})
#     clean_df.insert(0, "Book #", BOOK_NUMBERS.reset_index(drop=True))
#     return clean_df


# TODO: Map Roget to WordNet then back?
# TODO: What percentage of wordnet words are in roget?
class RogetMapper(FeatureReducer):
    def __init__(self, model: str = "", first_1k: bool = False, by_genre: bool = False, jupyter: bool = False,
                 model_loader: Optional[ModelLoader] = None, estimator: str = "svm", **kwargs):
        super().__init__(model=model, first_1k=first_1k, by_genre=by_genre, jupyter=jupyter, model_loader=model_loader, estimator=estimator,
                         mapping=True, **kwargs)
        self.model_loader.data = pd.merge(self.model_loader.data,
                                          model_loader.data[["Book #", "@Genre", "@Outcome", "@Downloads"]],
                                          on=["Book #", "@Genre"]).drop(columns=["@Outcome_x"]).rename(columns={"@Outcome_y": "@Outcome"})
        # self.model_loader.data = self.model_loader.data.drop(columns=["@Outcome_x", "@Downloads_x"])\
        #     .rename(columns={"@Outcome_y": "@Outcome", "@Downloads_y": "@Downloads"})
        self.model_loader.data = self.model_loader.data.loc[self.model_loader.data["@Genre"] != "Horror"]
        self.roget_thesaurus_df = invert_roget()

    def map_concat_test(self, map_from: str, map_to: str, scale: bool = False, data_to_map: Optional[pd.DataFrame] = None,
                        reduced_weights: Optional[Dict] = None, genre_list: List = NO_HORROR):
        """

        :param map_from:
        :param map_to:
        :param scale:
        :param genre_list:
        :param data_to_map:
        :param reduced_weights:
        :return:
        """
        nums_outcomes, to_map = self.get_to_map(self.model_loader.data if data_to_map is None else data_to_map, reduced_weights, genre_list)
        mapped_dict = self.map_to_roget(to_map, map_from, map_to, nums_outcomes, scale, genre_list)
        mapped = self.concat_map_to_roget(mapped_dict, map_to, nums_outcomes, scale, genre_list)

        mapped_df = self.concat_reorder(mapped)
        mapped_df_scaled = self.concat_reorder(mapped)

        accs, weights, preds = self.test_map_to_roget(mapped_df_scaled, map_to, genre_list)
        return accs, weights, preds, (mapped_df, mapped_df_scaled)

    def map_to_roget_simple(self, map_from: str, map_to: str, scale: bool = False, data_to_map: Optional[pd.DataFrame] = None,
                            reduced_weights: Optional[Dict] = None, genre_list: List = NO_HORROR):
        """

        :param map_from:
        :param map_to:
        :param scale:
        :param genre_list:
        :param data_to_map:
        :param reduced_weights:
        :return:
        """
        nums_outcomes, to_map = self.get_to_map(self.model_loader.data if data_to_map is None else data_to_map, reduced_weights, genre_list)
        mapped_dict = self.map_to_roget(to_map, map_from, map_to, nums_outcomes, scale, genre_list)
        return mapped_dict

    def get_to_map(self, data_to_map: pd.DataFrame, reduced_weights: Optional[Dict] = None, genre_list: List = NO_HORROR):
        """

        :param data_to_map:
        :param reduced_weights:
        :param genre_list:
        :return:
        """
        if self.by_genre:
            group_by_genre = data_to_map.groupby(by=["@Genre"], as_index=False)

            if reduced_weights is None:
                to_map = {genre: group.copy() for genre, group in group_by_genre}
            else:
                cols = {genre: ["Book #", "@Genre"] + list(reduced_weights[genre]["Feature"]) + ["@Outcome", "@Downloads"] for genre in genre_list}
                to_map = {genre: group[cols[genre]].copy() for genre, group in group_by_genre}
            nums_outcomes = {genre: group[["Book #", "@Genre", "@Outcome", "@Downloads"]].copy() for genre, group in group_by_genre}

        else:
            if reduced_weights is None:
                to_map = data_to_map.copy()
            else:
                cols = ["Book #", "@Genre"] + list(reduced_weights["Feature"]) + ["@Outcome", "@Downloads"]
                to_map = data_to_map[cols].copy()
            nums_outcomes = to_map[["Book #", "@Genre", "@Outcome", "@Downloads"]].copy()

        return nums_outcomes, to_map

    @staticmethod
    def concat_reorder(mapped: Union[Dict, pd.DataFrame]):
        if isinstance(mapped, dict):
            mapped_df = pd.concat(list(mapped.values())).fillna(0)
        else:
            mapped_df = mapped
        if mapped_df.columns[-1] != "@Downloads":
            if "@Downloads" not in mapped_df.columns:
                raise RuntimeError("@Downloads not in mapped_df.columns")
            out = mapped_df["@Downloads"]
            mapped_df.drop(columns=["@Downloads"], inplace=True)
            mapped_df["@Downloads"] = out
        if mapped_df.columns[-2] != "@Outcome":
            if "@Outcome" not in mapped_df.columns:
                raise RuntimeError("@Outcome not in mapped_df.columns")
            out = mapped_df["@Outcome"]
            mapped_df.drop(columns=["@Outcome"], inplace=True)
            mapped_df["@Outcome"] = out
        return mapped_df

    def map_to_roget(self, to_map: Union[Dict, pd.DataFrame], map_from: str, map_to: str, nums_outcomes: Union[Dict, pd.DataFrame],
                     scale: bool = False, genre_list: List = NO_HORROR, concat: bool = False, **kwargs):
        """

        :param to_map:
        :param map_from:
        :param map_to:
        :param nums_outcomes:
        :param scale:
        :param genre_list:
        :param concat:
        :param push:
        :return:
        """
        if map_from != "Word" and map_from != "Category" and map_from != "Section":
            raise RuntimeError(f"{map_from} is an invalid option for map_from.")
        if map_to != "Category" and map_to != "Section" and map_to != "Class":
            raise RuntimeError(f"{map_to} is an invalid option for map_to.")

        message = f"Mapping {self.model} to Roget {map_to}{' by genre' if self.by_genre else ''}"
        if self.jupyter:
            display(HTML(f"<h4>{message}</h4>"))
        else:
            print(message)

        if self.by_genre:
            return self._map_to_roget_by_genre(to_map, map_from, map_to, nums_outcomes, scale, genre_list, concat, **kwargs)
        else:
            return self._map_to_roget(to_map, map_from, map_to, nums_outcomes, scale, concat, **kwargs)

    def _map_to_roget_by_genre(self, dfs_to_map: Dict, map_from: str, map_to: str, nums_outcomes: Dict,
                               scale: bool = False, genre_list: List = NO_HORROR, concat: bool = False, **kwargs):
        """

        :param dfs_to_map:
        :param map_from:
        :param map_to:
        :param nums_outcomes:
        :param scale:
        :param genre_list:
        :param concat:
        :return:
        """
        mapped_dict = {genre: defaultdict(lambda: object) for genre in genre_list}
        bar_length = sum(len(dfs_to_map[genre].columns) - 4 for genre in genre_list)

        with self.tqdm(total=bar_length) as pbar:
            for genre in genre_list:
                pbar.set_postfix_str(f"-- {genre}")
                mapping_cols = dfs_to_map[genre].drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]).columns

                for col in mapping_cols:
                    # if push:
                    #     roget_map = self.roget_thesaurus_df[self.roget_thesaurus_df["Word"] == col][map_to]
                    # else:
                    roget_map = self.roget_thesaurus_df[self.roget_thesaurus_df[map_from] == col][map_to]

                    if map_to == "Section" or map_to == "Class":
                        roget_map = roget_map.unique()

                    for mapping in roget_map:
                        if mapping in mapped_dict[genre].keys():
                            mapped_dict[genre][mapping] = pd.concat([mapped_dict[genre][mapping], dfs_to_map[genre][col]], axis=1)
                        else:
                            mapped_dict[genre][mapping] = dfs_to_map[genre][[col]]
                    pbar.update(1)

        return self._concat_map_to_roget_by_genre(mapped_dict, map_to, nums_outcomes, scale, genre_list) if concat else mapped_dict

    def concat_map_to_roget(self, to_roget: Union[Dict, pd.DataFrame], map_to: str, nums_outcomes: Union[Dict, pd.DataFrame],
                            scale: bool = False, genre_list: List = NO_HORROR):
        """

        :param to_roget:
        :param map_to:
        :param nums_outcomes:
        :param scale:
        :param genre_list:
        :return:
        """
        display(HTML(f"<h4>Concatenating {self.model} to Roget {map_to}</h4>"))  # -- no scaling...</h4>"))
        if self.by_genre:
            return self._concat_map_to_roget_by_genre(to_roget, map_to, nums_outcomes, scale, genre_list)
        else:
            return self._concat_map_to_roget(to_roget, nums_outcomes, scale)

    def _concat_map_to_roget_by_genre(self, to_roget: Dict, map_to: str, nums_outcomes: Dict, scale: bool = False, genre_list: List = NO_HORROR):
        concat = {}
        with self.tqdm(total=len(genre_list)) as pbar:
            for genre in genre_list:
                pbar.set_postfix_str(f" -- {genre}")
                to_concat = [pd.DataFrame({k: to_roget[genre][k].sum(axis=1)}) for k in to_roget[genre].keys()]
                mapped = pd.concat(to_concat, axis=1)
                mapped.insert(0, "@Genre", nums_outcomes[genre]["@Genre"])
                mapped.insert(0, "Book #", nums_outcomes[genre]["Book #"])
                mapped["@Outcome"] = nums_outcomes[genre]["@Outcome"]
                mapped["@Downloads"] = nums_outcomes[genre]["@Downloads"]
                if scale and self.by_genre:
                    mapped, _ = process_and_scale(mapped, self.model)
                concat[genre] = mapped
                pbar.update(1)

        if scale and not self.by_genre:
            concat_df = self.scale_mapped(concat, map_to)
            return concat_df
        # scaled = {}
        # display(HTML(f"<h4>Concatenating {self.model} to Roget {map_to} -- scaling by genre...</h4>"))
        # with self.tqdm(total=len(genre_list)) as pbar:
        #     for genre in genre_list:
        #         mapped = self.concat_map(genre, map_to_roget_dict, nums_outcomes, pbar)
        #         scaled[genre], _ = process_and_scale(mapped, self.model)
        #         pbar.update(1)

        # return no_scale, scaled
        return concat

    # def concat_map(self, genre: str, map_to_roget_dict: Dict, nums_outcomes: Dict, pbar):
    #     pbar.set_postfix_str(f" -- {genre}")
    #     mapped = pd.concat([pd.DataFrame({k: map_to_roget_dict[genre][k].sum(axis=1).reset_index(drop=True)}) for k in map_to_roget_dict[genre].keys()], axis=1)
    #     mapped.insert(0, "@Genre", nums_outcomes[genre]["@Genre"])
    #     mapped.insert(0, "Book #", nums_outcomes[genre]["Book #"])
    #     mapped["@Outcome"] = nums_outcomes[genre]["@Outcome"]
    #     mapped, _ = process_and_scale(mapped, self.model)
    #     return mapped

    def _map_to_roget(self, df_to_map: pd.DataFrame, map_from: str, map_to: str,
                      nums_outcomes: pd.DataFrame, scale: bool = False, concat: bool = False, **kwargs):
        """

        :param df_to_map:
        :param map_from:
        :param map_to:
        :param nums_outcomes:
        :param scale:
        :param concat:
        :return:
        """
        mapped_dict = defaultdict(lambda: pd.DataFrame())
        mapping_cols = df_to_map.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]).columns

        for col in self.tqdm(mapping_cols, postfix=f"-- MAPPING TO ROGET {map_to.upper()}"):
            # if push:
            #     roget_map = self.roget_thesaurus_df[self.roget_thesaurus_df["Word"] == col][map_to]
            # else:
            roget_map = self.roget_thesaurus_df[self.roget_thesaurus_df[map_from] == col][map_to]

            if map_to == "Section" or map_to == "Class":
                roget_map = roget_map.unique()

            for mapping in roget_map:
                if mapping in mapped_dict.keys():
                    mapped_dict[mapping] = pd.concat([mapped_dict[mapping], df_to_map[col]], axis=1)
                else:
                    mapped_dict[mapping] = df_to_map[[col]]

        return self._concat_map_to_roget(mapped_dict, nums_outcomes, scale) if concat else mapped_dict

    def _concat_map_to_roget(self, to_roget: Dict, nums_outcomes: pd.DataFrame, scale: bool = False):
        to_concat = [pd.DataFrame({k: data.sum(axis=1)}) for k, data in to_roget.items()]
        mapped = pd.concat(to_concat, axis=1)
        mapped.insert(0, "@Genre", nums_outcomes["@Genre"])
        mapped.insert(0, "Book #", nums_outcomes["Book #"])
        mapped["@Outcome"] = nums_outcomes["@Outcome"]
        mapped["@Downloads"] = nums_outcomes["@Downloads"]
        if scale:
            mapped, _ = process_and_scale(mapped, self.model)
        return mapped

    def test_map_to_roget(self, mapped: Union[Dict, pd.DataFrame], map_to: str, genre_list: List = NO_HORROR, g_predict: Optional[str] = None):
        display(HTML(f"<h4>Testing {self.model} to Roget {map_to}</h4>"))  # -- All Genres Scaled</h4>"))

        if isinstance(mapped, dict):
            mapped_df = pd.DataFrame(list(mapped.values())).fillna(0)
        else:
            mapped_df = mapped.copy()
        # full_map_to_roget_scaled, _ = process_and_scale(full_map_to_roget, self.model)

        # if g_predict is not None:
        #     scaled = pd.concat(list(scaled_.values())).fillna(0)
        #     full_map_to_roget_acc, full_map_to_roget_weights = self.predict_genre(how=g_predict, genre_list=genre_list,
        #                                                                           disp_acc=False, disp_weights=False,
        #                                                                           show_pbar=False)
        # else:

        if self.by_genre:
            map_to_roget_acc, map_to_roget_weights, map_to_roget_preds = self.predict_by_genre(genre_list=genre_list, temp=mapped_df,
                                                                                               disp_acc=False, disp_weights=False)
        else:
            map_to_roget_acc, map_to_roget_weights, map_to_roget_preds = self.predict_all(temp=mapped_df, disp_acc=False, disp_weights=False)

        map_to_roget_acc = map_to_roget_acc[map_to_roget_acc["Genre"] != "Average"]
        map_to_roget_acc = map_to_roget_acc.append({"Genre": "Average", "Accuracy": map_to_roget_acc["Accuracy"].mean()}, ignore_index=True)
        display_df(map_to_roget_acc)

        # display(HTML(f"<h4>Testing {self.model} to Roget {map_to} -- Scaled By Genre</h4>"))
        # map_to_roget_results = []
        # map_to_roget_weights = {}
        #
        # for genre in genre_list:
        #     if g_predict is not None:
        #         acc, weights = self.predict_genre(how=g_predict, genre_list=[genre], searching=True, disp_acc=False,
        #                                           disp_weights=False, show_pbar=False)
        #     else:
        #         acc, weights = self.predict_success_by_genre(genre_list=[genre], searching=True, disp_acc=False,
        #                                                      disp_weights=False, show_pbar=False)
        #
        #     acc = acc[acc["Genre"] != "Average"]
        #     map_to_roget_results.append(acc)
        #     map_to_roget_weights.update(weights)
        #
        # map_to_roget_acc = pd.concat(map_to_roget_results)
        # map_to_roget_acc = map_to_roget_acc.append({"Genre": "Average", "Accuracy": map_to_roget_acc["Accuracy"].mean()}, ignore_index=True)
        # display_df(map_to_roget_acc)

        return map_to_roget_acc, map_to_roget_weights, map_to_roget_preds

    def scale_mapped(self, mapped: Dict, map_to: str):
        df = pd.concat(list(mapped.values())).fillna(0)
        df_scaled, _ = process_and_scale(df, f"{self.model} to roget {map_to}")
        return df_scaled

    def reduce_features_by_genre(self, model_weights: Dict, og_acc: pd.DataFrame, og_preds: Dict, data: Optional[pd.DataFrame] = None,
                                 genre_list: List[str] = NO_HORROR, g_predict: List = None, **kwargs):
        self.model_loader.data = data
        return super().reduce_features_by_genre(model_weights, og_acc, og_preds, data, genre_list, g_predict, **kwargs)

    def print_num_features(self, df: Optional[pd.DataFrame] = None):
        for genre, data in df.groupby(by="@Genre", as_index=False):
            print(genre, filter_out_zeros(data.drop(columns=["Book #", "@Genre", "@Outcome"] + (["@Downloads"] if "@Downloads" in data.columns else []))).shape)
