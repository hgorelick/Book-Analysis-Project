import pickle
from collections import Counter

import pandas as pd
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import cufflinks as cf
from typing import List

from analysis.thresholds import ThresholdApplier
from book_processor.gutenberg_processor import process_rdf_df
from loading_utils.data_loader import DataLoader
from notebook_utils.constants import PROJ_ROOT, NEW_GENRES, scaler
from notebook_utils.utils import display_df

cf.go_offline(connected=True)
init_notebook_mode(connected=True)


def get_general_stats(by_genre: bool = False, jupyter: bool = False):
    ta = ThresholdApplier("clausal", by_genre=by_genre, jupyter=jupyter)
    data_df = ta.merge_with_downloads()
    genre_groups = data_df.loc[data_df["@Genre"] != "(Auto)Biography"].groupby(by=["@Genre"])

    stats_df = genre_groups.describe()[["Downloads"]]
    if jupyter:
        display_df(stats_df, index=True)

    keep = ["Book #", "@Genre", "Downloads"]
    return data_df.drop(columns=[col for col in data_df.columns if col not in keep])


def plot_stats(df: pd.DataFrame, genres: List[str], stat: str, bins: int):
    if genres[0] == "All":
        grouped = df.sort_values(by="Downloads", ascending=False).groupby(by=["@Genre"])
        data = []
        for genre, group in grouped:
            trace = go.Histogram(x=group["Downloads"], nbinsx=bins, name=genre)
            data.append(trace)
        fig = go.Figure(data)
        iplot(fig)
    else:
        grouped = df.loc[df["@Genre"].isin(genres)].sort_values(by="Downloads", ascending=False).groupby(by=["@Genre"])
        data = []

        for genre, group in grouped:
            # group["Downloads"] = scale.fit_transform(group[["Downloads"]])
            if stat == "All":
                trace = go.Histogram(x=group["Downloads"], nbinsx=bins, name=genre)
            elif stat == "Count":
                trace = go.Bar(x=len(group), name=genre)
            elif stat == "Mean":
                trace = go.Bar(x=group["Downloads"].mean(), name=genre)
            elif stat == "Median":
                trace = go.Bar(x=group["Downloads"].median(), name=genre)
            elif stat == "Max":
                trace = go.Bar(x=group["Downloads"].max(), name=genre)
            elif stat == "Min":
                trace = go.Bar(x=group["Downloads"].min(), name=genre)
            else:
                raise ValueError("Invalid stat given.")
            data.append(trace)
        fig = go.Figure(data)
        iplot(fig)


def distribute_genres():
    finished_books = DataLoader().init_mined
    rdf_data_df, by_genre_dict = process_rdf_df()
    rdf_data_df = rdf_data_df.loc[rdf_data_df["Genre"] != "(Auto)Biography"]
    rdf_data_df = rdf_data_df.loc[rdf_data_df["Book #"].isin(finished_books)]

    genre_lengths = {name: len(group) for name, group in rdf_data_df.groupby(by=["Genre"])}
    genre_lengths_df = pd.DataFrame({"Genre": list(genre_lengths.keys()), "Length": list(genre_lengths.values())}).sort_values(by=["Length"], ascending=False)
    rdf_data_df = pd.merge(rdf_data_df, genre_lengths_df, how="outer", on="Genre")

    groupby_title = rdf_data_df.groupby(by=["Title"])
    title_groups = [group for name, group in groupby_title if len(group) > 1]  # and any(group["Genre"] == "Fiction")]

    indexes = []
    for group in title_groups:
        for idx, data in group.loc[group["Length"] != group["Length"].min()].iterrows():
            indexes.append(idx)
            genre_lengths[data["Genre"]] -= 1
        # genre_lengths[group.loc[group["Length"] == group["Length"].min(), "Genre"]]

    indexes = list(set(indexes))
    indexes.sort()
    changed = rdf_data_df.copy().drop(indexes)
    changed_groupby_genre = changed.groupby(by=["Genre"])
    changed_genre_groups = [group for name, group in changed_groupby_genre]
    changed_genre_groups.sort(key=len, reverse=True)
    genres = pd.concat(changed_genre_groups).drop(columns=["Length"]).sort_values(by=["Genre"]).reset_index(drop=True)

    with open(str(PROJ_ROOT.joinpath("data", "rdf_data", "processed rdf data")), "wb+") as f:
        pickle.dump(genres, f, protocol=4)

    return genres


# TODO: Get b#s of books that are going to change genres --> after loading the data, change the genres of those books
def books_with_multiple_genres():
    finished_books = DataLoader().init_mined
    rdf_data_df, by_genre_dict = process_rdf_df()
    rdf_data_df = rdf_data_df.loc[rdf_data_df["Genre"] != "(Auto)Biography"]
    rdf_data_df = rdf_data_df.loc[rdf_data_df["Book #"].isin(finished_books)]

    groupby_title = rdf_data_df.groupby(by=["Title"])
    title_groups = [group for name, group in groupby_title if len(group) > 1]  # and any(group["Genre"] == "Fiction")]
    groupby_genre = rdf_data_df.groupby(by=["Genre"])

    genre_groups_dict = {name: group for name, group in groupby_genre}
    genre_groups = list(genre_groups_dict.values())
    genre_groups.sort(key=len, reverse=True)

    changed_genres = {}
    indexes = []
    i = 0
    while i < len(genre_groups) // 2:
        j = len(genre_groups) - 1
        while j > len(genre_groups) // 2:
            merged = pd.merge(genre_groups[i], genre_groups[j], on=["Book #", "Title"])
            merged.drop(columns=[col for col in merged.columns if "Lang" in col or "Auth" in col or "Subj" in col], inplace=True)
            for row, data in merged.iterrows():
                # if data["Book #"] not in changed_genres:
                changed_genres[data["Book #"]] = (data["Genre_x"], data["Genre_y"])
                matches = rdf_data_df.loc[(rdf_data_df["Book #"] == data["Book #"]) &
                                          (rdf_data_df["Title"] == data["Title"]) &
                                          (rdf_data_df["Downloads"] == data["Downloads_x"])]
                # count = 0
                for idx in matches.index.values:
                    if rdf_data_df.loc[(rdf_data_df["Book #"] == data["Book #"]) &
                                       (rdf_data_df["Title"] == data["Title"]) &
                                       (rdf_data_df["Genre"] == data["Genre_x"]) &
                                       (rdf_data_df["Downloads"] == data["Downloads_x"]), "Genre"].values[0] != data["Genre_y"]:
                        if idx not in indexes:
                            indexes.append(idx)
                        # count += 1
                        # if count == len(matches) - 1:
                        #     break
                # indexes.append(rdf_data_df.loc[(rdf_data_df["Book #"] == data["Book #"]) &
                #                                (rdf_data_df["Title"] == data["Title"]) &
                #                                (rdf_data_df["Genre"] == data["Genre_x"]) &
                #                                (rdf_data_df["Downloads"] == data["Downloads_x"])].index.values[0])
            j -= 1
        i += 1

    change_counts = Counter(changed_genres.values())

    indexes = list(set(indexes))
    indexes.sort()
    # to_drop = rdf_data_df.iloc[indexes, :]
    changed = rdf_data_df.copy().drop(indexes)

    # comp_genres = [(g1, g2) for g1, g2 in combinations(genre_groups, 2)]
    #
    # multi = {(g1, g2): 0 for g1, g2 in combinations(NEW_GENRES, 2)}
    # for name, group in tqdm(groupby_title, postfix="-- TITLE GROUPS"):
    #     genres = list(set(list(group["Genre"].values)))
    #     genres.sort()
    #     if len(genres) > 1:
    #         for g1, g2 in combinations(genres, 2):
    #             try:
    #                 multi[(g1, g2)] += 1
    #             except KeyError:
    #                 print(group["Genre"])

    # return multi

    changed_groupby_genre = changed.groupby(by=["Genre"])
    changed_genre_groups = {name: group for name, group in changed_groupby_genre}
    return changed_genre_groups


if __name__ == "__main__":
    data_df = get_general_stats(by_genre=True, jupyter=True)
    plot_stats(data_df, ["All"], "", 10)
