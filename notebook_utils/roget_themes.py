__all__ = ['get_themes_by_genre', 'plot_tbg', 'get_sorted_tbg', 'get_rosect_freq', 'get_wn_freq', 'plot_theme_freq_diff_vs_weight',
           'plot_themes_by_genre', 'get_theme_diffs', 'plot_theme_diffs', 'plot_avg_freq_diff_by_genre', 'get_freq_weight_ranks',
           'get_rankings', 'search_rank_bins']

from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd

from IPython.display import display, HTML
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tqdm.notebook import tqdm_notebook as tqdm

from notebook_utils.constants import NO_HORROR, scaler, PROJ_ROOT
from notebook_utils.utils import display_df, setup_axis, setup_markers

tqdm.pandas()


# TODO: See how each book of each genre fits (rank correlation) to its theme graph
#           - Compare that fit to the downloads and rank and plot

def get_themes_by_genre(themes_df: pd.DataFrame, sect_weights: Dict, genre_list: List = NO_HORROR):
    themes_by_genre = []
    bar_length = len(themes_df.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]).columns) * len(genre_list)
    themes = list(themes_df.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]).columns)
    themes.sort()

    display(HTML("<h4>Getting themes by genre...</h4>"))
    with tqdm(total=bar_length) as pbar:
        for genre in genre_list:
            pbar.set_postfix_str(f" -- {genre}")
            for theme in themes:
                try:
                    theme_weight = sect_weights[genre].set_index("Feature").loc[theme, "Weight"]
                except KeyError:
                    theme_weight = 0
                themes_by_genre.append({"Genre": genre, "Theme": theme, "Weight": theme_weight})
                pbar.update(1)

    tbg_df = pd.DataFrame(themes_by_genre)
    tbg_df.loc[tbg_df["Theme"] == theme, "Weight"] = tbg_df.loc[tbg_df["Theme"] == theme, "Weight"].abs()
    # for theme in themes:
    #     tbg_weights_scaled = scale.fit_transform(tbg_df.loc[tbg_df["Theme"] == theme][["Weight"]].abs())
    #     tbg_df.loc[tbg_df["Theme"] == theme, "Weight"] = tbg_weights_scaled

    return tbg_df, themes


def plot_tbg(tbg_df: pd.DataFrame, themes: List, colors: Dict, sort: bool = False, scatter: bool = False):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

    if sort:
        sorted_tbg_df = get_sorted_tbg(tbg_df, colors)
        for i in range(len(sorted_tbg_df["Theme"])):
            axes.bar(sorted_tbg_df["Theme"][i], sorted_tbg_df["percentage"][i], width=0.5, color=sorted_tbg_df["color"][i],
                     bottom=sorted_tbg_df["bottoms"][i], label=sorted_tbg_df["Genre"][i])

        plt.xticks(rotation=90)

    else:
        if scatter:
            for genre, color in zip(NO_HORROR, list(colors.values())):
                sizes = tbg_df[tbg_df["Genre"] == genre].sort_values(by=["Theme"])["Weight"] * 2 ** 12
                tbg_df[tbg_df["Genre"] == genre].sort_values(by=["Theme"]).plot(x="Theme", y="Weight", ax=axes, kind="scatter", s=sizes, rot=90,
                                                                                c=color, alpha=0.9)
        else:
            tbg_percentage = tbg_df.copy()
            for theme in themes:
                theme_sum = tbg_percentage.loc[tbg_percentage["Theme"] == theme, "Weight"].sum()
                tbg_percentage.loc[tbg_percentage["Theme"] == theme, "Weight"] /= theme_sum

            margin_bottom = np.zeros(len(tbg_percentage["Theme"].drop_duplicates()))

            for genre, color in zip(NO_HORROR, list(colors.values())):
                values = list(tbg_percentage[tbg_percentage["Genre"] == genre].sort_values(by=["Theme"]).loc[:, "Weight"])
                tbg_percentage[tbg_percentage["Genre"] == genre].sort_values(by=["Theme"]).plot.bar(x="Theme", y="Weight", ax=axes, stacked=True,
                                                                                                    width=0.5,
                                                                                                    bottom=margin_bottom, rot=90, color=color)
                margin_bottom += values

    if sort:
        setup_axis(axes, xmin=None, ymajor=10, ylabel="Weight Percentage")
    else:
        setup_axis(axes, ymajor=10, x_ticklabels=themes, ylabel="Weight Percentage" if not scatter else "Weight",
                   bottom=-0.01 if scatter else None, top=1.05 if scatter else None)

    if sort:
        axes.legend(bbox_to_anchor=(0.9915, 1.07), fontsize=19, ncol=len(NO_HORROR))
        handles, labels = axes.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        sorted_keys = sorted(by_label)
        sorted_vals = [by_label[k] for k in sorted_keys]
        by_label = dict(zip(sorted_keys, sorted_vals))
        axes.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.9915, 1.07), fontsize=19, ncol=len(NO_HORROR))
    elif scatter:
        legend = axes.legend(NO_HORROR, bbox_to_anchor=(0.9915, 1.07), fontsize=19, ncol=len(NO_HORROR))
        for i in range(len(legend.legendHandles)):
            legend.legendHandles[i]._sizes = [250]
    else:
        axes.legend(NO_HORROR, bbox_to_anchor=(0.9915, 1.07), fontsize=19, ncol=len(NO_HORROR))

    plt.margins(x=0.025, y=0.05)
    plt.show()


def get_sorted_tbg(tbg_df: pd.DataFrame, colors: Dict):
    s_tbg_df = tbg_df.copy()

    s_tbg_df["percentage"] = s_tbg_df["Weight"] / s_tbg_df.groupby("Theme")["Weight"].transform("sum")
    s_tbg_df.sort_values("percentage", ascending=False, inplace=True)

    s_tbg_df = s_tbg_df.groupby("Theme").apply(ranker)
    s_tbg_df.sort_values(["Theme", "rank"], ascending=[True, True], inplace=True)

    s_tbg_df["color"] = s_tbg_df.apply(color_assigment, args=(colors,), axis=1)

    s_tbg_df["bottoms"] = s_tbg_df.groupby("Theme")["percentage"].cumsum() - s_tbg_df["percentage"]
    s_tbg_df["Theme"] = s_tbg_df["Theme"].astype(str)

    s_tbg_df = s_tbg_df.reset_index(drop=True)
    return s_tbg_df


def ranker(df: pd.DataFrame):
    df["rank"] = np.arange(len(df)) + 1
    return df


def color_assigment(df: pd.DataFrame, colors: Dict):
    return colors[df["Genre"]]


def get_rosect_freq(theme_df: pd.DataFrame, themes: List, sect_weights: Dict, estimator: str = "svm", scale_weights: bool = True,
                    genre_list: List = NO_HORROR, g_predict: Optional[str] = None):
    map_to_rosect_wvs = {genre: [] for genre in genre_list}
    display(HTML("<h4>Calculating Roget Section Frequency by Success per Genre</h4>"))
    with tqdm(total=len(genre_list)) as pbar:
        for genre in genre_list:
            pbar.set_postfix_str(f" -- {genre}")

            for theme in themes:
                try:
                    theme_avg_freq1, theme_avg_freq2 = calculate_theme_freq(theme_df, genre, theme, estimator, g_predict)
                except KeyError as e:
                    theme_df[genre][theme] = 0
                    theme_avg_freq1, theme_avg_freq2 = calculate_theme_freq(theme_df, genre, theme, estimator, g_predict)

                try:
                    weight = abs(sect_weights[genre].set_index("Feature").loc[theme, "Weight"])
                except KeyError:
                    weight = 0

                freq_diff = theme_avg_freq1 - theme_avg_freq2
                freq_label = "Frequency Difference" if estimator == "svm" else "Frequency"

                map_to_rosect_wvs[genre].append({"Genre": genre, "Theme": theme, f"{freq_label}": freq_diff, "Weight": weight})

            map_to_rosect_wvs[genre] = pd.DataFrame(map_to_rosect_wvs[genre]).sort_values(by=["Theme"])

            # if name != "":
            #     write_wvs_csv(map_to_rosect_wvs[genre], name, genre)
            # print(f"Before filter:\n{map_to_rosect_wvs[genre]}")
            map_to_rosect_wvs[genre] = map_to_rosect_wvs[genre][(map_to_rosect_wvs[genre][freq_label] != 0) |
                                                                (map_to_rosect_wvs[genre]["Weight"] != 0)].reset_index(drop=True)
            # print(f"After filter:\n{map_to_rosect_wvs[genre]}")
            pbar.update(1)

    return scale_mapped_weights(map_to_rosect_wvs, genre_list) if scale_weights else map_to_rosect_wvs


def scale_mapped_weights(map_to_rosect_wvs: Dict, genre_list: List[str] = NO_HORROR):
    for genre in genre_list:
        map_to_rosect_wvs_scaled = scaler.fit_transform(map_to_rosect_wvs[genre].loc[map_to_rosect_wvs[genre]["Genre"] == genre][["Weight"]])
        map_to_rosect_wvs[genre].loc[map_to_rosect_wvs[genre]["Genre"] == genre, "Weight"] = map_to_rosect_wvs_scaled
    return map_to_rosect_wvs


def calculate_theme_freq(theme_df: pd.DataFrame, genre: str, theme: str, estimator: str = "svm", g_predict: Optional[str] = None):
    if g_predict == "one_v_one":
        theme_avg_freq1 = theme_df.loc[theme_df["@Genre"] == genre[0], theme].mean()
        theme_avg_freq2 = theme_df.loc[theme_df["@Genre"] == genre[1], theme].mean()
    elif g_predict == "one_v_all":
        # ova = pd.concat(list(scaled.values())).fillna(0)
        theme_avg_freq1 = theme_df.loc[theme_df["@Genre"] == genre, theme].mean()
        theme_avg_freq2 = theme_df.loc[theme_df["@Genre"] != genre, theme].mean()
    else:
        if estimator == "svm":
            theme_avg_freq1 = theme_df.loc[(theme_df["@Genre"] == genre) & (theme_df["@Outcome"] == "SUCCESSFUL"), theme].mean()
            theme_avg_freq2 = theme_df.loc[(theme_df["@Genre"] == genre) & (theme_df["@Outcome"] == "FAILURE"), theme].mean()
        else:
            theme_avg_freq1 = theme_df.loc[theme_df["@Genre"] == genre, theme].mean()
            theme_avg_freq2 = 0
    return theme_avg_freq1, theme_avg_freq2


def get_freq_weight_ranks(mapped: Dict, col: str, disp: bool = True):
    rankings = {}
    for genre, data in mapped.items():
        ranked = data.loc[data["Weight"] != 0].copy()
        ranked = ranked.sort_values(by="Weight", ascending=False).copy().reset_index(drop=True).reset_index().rename(columns={"index": "Weight Rank"})
        ranked = ranked.sort_values(by="Frequency Difference", ascending=False).reset_index(drop=True).reset_index().rename(columns={"index": "Freq Rank"})
        ranked = ranked[["Genre", "Theme", "Freq Rank", "Weight Rank"]].sort_values(by="Theme")
        rankings[genre] = ranked

    rankings_df = pd.concat(list(rankings.values())).fillna(0)
    # print(rankings_df)

    for theme, data in rankings_df.groupby(by="Theme", as_index=False):
        for genre in NO_HORROR:
            if theme not in rankings[genre]["Theme"].values:
                row = {"Genre": genre, "Theme": theme, "Freq Rank": None, "Weight Rank": None}
                rankings[genre] = rankings[genre].append(row, ignore_index=True)
            rankings[genre].sort_values(by=["Theme"], inplace=True)

    rankings_df = pd.concat(list(rankings.values()))

    if disp:
        for genre, data in rankings_df.groupby(by="Genre", as_index=False):
            display_df(data.sort_values(by=f"{col} Rank"))

    return rankings_df


def get_rankings(preds: Dict):
    rankings = {}
    for genre, data in preds.items():
        ranked = preds[genre].copy()
        ranked["Pred"] = scaler.fit_transform(ranked[["Pred"]])
        # ranked["Outcome Rank"] = ranked.sort_values(by="@Outcome", ascending=False).reset_index()["index"]
        ranked["Outcome Rank"] = scaler.fit_transform(ranked.sort_values(by="@Outcome", ascending=False).reset_index()[["index"]])
        # ranked["Pred Rank"] = ranked.sort_values(by="Pred", ascending=False).reset_index()["index"]
        ranked["Pred Rank"] = scaler.fit_transform(ranked.sort_values(by="Pred", ascending=False).reset_index()[["index"]])
        rankings[genre] = ranked
    return rankings


def search_rank_bins(rankings: Dict, jupyter: bool = False):
    if jupyter:
        tq = globals()["tqdm"]
    else:
        from tqdm import tqdm
        tq = tqdm

    bin_results = []
    for genre, data in rankings.items():
        best_acc = 1.0 - mean_squared_error(data["Outcome Rank"], data["Pred Rank"])
        # print(best_acc)
        bin_results.append({"Genre": genre, "Bins": len(data), "Acc": best_acc})

        k = len(data) // 2
        i = 0
        with tq(total=len(data), postfix=f"SEARCHING BINS -- {genre}") as pbar:
            while True:
                if best_acc == 1.0:
                    print(f"{genre} predicted downloads rank bins exhausted at {k} bins (100%)")
                    pbar.update(len(data) - i)
                    break

                test_bins = data.copy()
                binned_out, binned_pred = fit_transform_bins(k, data)
                test_bins["Outcome Rank"] = binned_out
                test_bins["Pred Rank"] = binned_pred

                k_acc = 1.0 - mean_squared_error(test_bins["Outcome Rank"], test_bins["Pred Rank"])
                bin_results.append({"Genre": genre, "Bins": k, "Acc": k_acc})

                if k_acc >= best_acc:
                    best_acc = k_acc
                    k -= k // 2
                else:
                    k += k // 2

                if k < 2 or k > len(data):
                    print(f"{genre} predicted downloads rank bins exhausted at {k} bins (100%)")
                    pbar.update(len(data) - i)
                    break

                pbar.update(1)
                i += 1

    bin_results_df = pd.DataFrame(bin_results).sort_values(by=["Acc"], ascending=False)
    return bin_results_df


def fit_transform_bins(k: int, data: pd.DataFrame):
    kbin_out = KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="uniform")
    # kbin_out.fit(data[["Outcome Rank"]])
    kbin_pred = KBinsDiscretizer(n_bins=k, encode="ordinal", strategy="uniform")
    # kbin_pred.fit(data[["Pred Rank"]])
    binned_out = kbin_out.fit_transform(data[["Outcome Rank"]])
    binned_pred = kbin_pred.fit_transform(data[["Pred Rank"]])
    return binned_out, binned_pred


def get_wn_freq(wn_set: Dict, sect_weights: Dict, name: str = "", genre_list: List = NO_HORROR, g_predict: Optional[str] = None):
    wn_wvs = {genre: [] for genre in genre_list}

    display(HTML("<h4>Calculating WordNet Word Frequency by Success per Genre</h4>"))
    bar_length = sum(len(wn_set[genre].columns) - 3 for genre in genre_list)

    with tqdm(total=bar_length) as pbar:
        for genre in genre_list:
            pbar.set_postfix_str(f" -- {genre}")

            for col in wn_set[genre].drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"]).columns:
                try:
                    word_avg_freq1, word_avg_freq2 = calculate_word_freq(wn_set, genre, col, g_predict)
                except KeyError as e:
                    wn_set[genre][col] = 0
                    word_avg_freq1, word_avg_freq2 = calculate_word_freq(wn_set, genre, col, g_predict)

                try:
                    weight = abs(sect_weights[genre].set_index("Feature").loc[col, "Weight"])
                except KeyError:
                    weight = 0

                freq_diff = word_avg_freq1 - word_avg_freq2
                wn_wvs[genre].append({"Genre": genre, "Word": col, "Frequency Difference": freq_diff, "Weight": weight})
                pbar.update(1)

            wn_wvs[genre] = pd.DataFrame(wn_wvs[genre]).sort_values(by=["Word"])

            if name != "":
                write_wvs_csv(wn_wvs[genre], name, genre)
            # print(f"Before filter:\n{wn_wvs[genre]}")
            wn_wvs[genre] = wn_wvs[genre][(wn_wvs[genre]["Frequency Difference"] != 0) |
                                          (wn_wvs[genre]["Weight"] != 0)].reset_index(drop=True)
            # print(f"After filter:\n{wn_wvs[genre]}")

    return scale_mapped_weights(wn_wvs, genre_list)
    # for genre in genre_list:
    #     wn_weight_scaled = scale.fit_transform(wn_wvs[genre].loc[wn_wvs[genre]["Genre"] == genre][["Weight"]])
    #     wn_wvs[genre].loc[wn_wvs[genre]["Genre"] == genre, "Weight"] = wn_weight_scaled
    #
    # return wn_wvs


def calculate_word_freq(wn_set: Dict, genre: str, col: str, g_predict: Optional[str] = None):
    if g_predict == "one_v_one":
        word_avg_freq1 = wn_set[genre].loc[wn_set[genre]["@Genre"] == genre[0], col].mean()
        word_avg_freq2 = wn_set[genre].loc[wn_set[genre]["@Genre"] == genre[1], col].mean()
    elif g_predict == "one_v_all":
        ova = pd.concat(list(wn_set.values())).fillna(0)
        word_avg_freq1 = ova.loc[ova["@Genre"] == genre, col].mean()
        word_avg_freq2 = ova.loc[ova["@Genre"] != genre, col].mean()
    else:
        word_avg_freq1 = wn_set[genre].loc[wn_set[genre]["@Outcome"] == "SUCCESSFUL", col].mean()
        word_avg_freq2 = wn_set[genre].loc[wn_set[genre]["@Outcome"] == "FAILURE", col].mean()
    return word_avg_freq1, word_avg_freq2


def write_wvs_csv(wvs: pd.DataFrame, name: str, genre: str):
    wvs_csv = open(str(PROJ_ROOT.joinpath("data", genre, f"{genre}_{name}_wvs.csv")), 'w+', newline='')
    wvs.to_csv(wvs_csv, index=False)


def plot_theme_freq_diff_vs_weight(map_to_rosect_wvs: Dict, colors: Dict, other_wvs: Optional[Dict] = None, genre_list: Dict = NO_HORROR,
                                   common_only: bool = False):
    for genre in genre_list:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

        themes = list(map_to_rosect_wvs[genre]["Theme"])

        if other_wvs is not None and not common_only:
            themes = list(set(themes + list(other_wvs[genre]["Theme"])))

        themes.sort()
        legend_elems = [Line2D([0], [0], marker="o", color="white", markerfacecolor=colors[theme], label=theme, markersize=20) for theme in
                        themes]

        for theme in themes:
            if theme in list(map_to_rosect_wvs[genre]["Theme"]):
                map_to_rosect_wvs[genre][map_to_rosect_wvs[genre]["Theme"] == theme].plot(x="Weight", y="Frequency Difference", ax=axes,
                                                                                          linestyle="none",
                                                                                          marker="o", markersize=30, color=colors[theme],
                                                                                          alpha=0.9)
            if other_wvs is not None:
                fill = "full" if theme in list(map_to_rosect_wvs[genre]["Theme"]) else "none"
                a = 0.9 if fill == "full" else 1.0
                other_wvs[genre][other_wvs[genre]["Theme"] == theme].plot(x="Weight", y="Frequency Difference", ax=axes, linestyle="none",
                                                                          marker="D", markeredgewidth=3, fillstyle=fill, markersize=30,
                                                                          color=colors[theme], alpha=a)

        axes.set_title(genre, fontsize=32)

        top = max(map_to_rosect_wvs[genre]["Frequency Difference"].max(), abs(map_to_rosect_wvs[genre]["Frequency Difference"].min()))
        bottom = min(-map_to_rosect_wvs[genre]["Frequency Difference"].max(), map_to_rosect_wvs[genre]["Frequency Difference"].min())

        if map_to_rosect_wvs[genre]["Frequency Difference"].min() > -0.05:
            bottom = -0.05

        if other_wvs is not None:
            top = max(top, other_wvs[genre]["Frequency Difference"].max(), abs(other_wvs[genre]["Frequency Difference"].min()))
            bottom = min(bottom, other_wvs[genre]["Frequency Difference"].min())

        ymajor = 40 if top > 0.05 else 160
        offset = 0.02 if top > 0.05 else 0.002
        setup_axis(axes, ymin=-1, ymajor=ymajor, yminor=ymajor * 5,
                   x_ticklabel_size=22, xlabel="Weight", xlabel_size=32, xlabel_pad=30, ylabel="Avg Frequency Difference",
                   left=-0.025, right=1.025,
                   bottom=bottom - offset,
                   top=top + offset)

        axes.axhline(linestyle="--", linewidth=3, color="black", alpha=0.5)
        axes.legend(handles=legend_elems, bbox_to_anchor=(1.005, 0.95), loc="upper left", fontsize=22)

        plt.show()


def plot_themes_by_genre(map_to_rosect_wvs: Dict, colors: Dict, col: str, other_wvs: Optional[Dict] = None, genre_list: List = NO_HORROR,
                         estimator: str = "svm"):
    for genre in genre_list:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

        map_to_rosect_wvs[genre].sort_values(by=["Theme"], inplace=True)

        themes = list(map_to_rosect_wvs[genre]["Theme"])
        themes.sort()

        top = map_to_rosect_wvs[genre][col].max()
        bottom = map_to_rosect_wvs[genre][col].min()

        if other_wvs is not None:
            legend_elems = [Patch(facecolor="white", edgecolor="black", label=f"Reduced {col}")]
            themes = list(set(themes + list(other_wvs[genre]["Theme"])))
            positions = np.arange(0, len(themes))

            merged = pd.merge(map_to_rosect_wvs[genre], other_wvs[genre], on=["Genre", "Theme"], how="outer").fillna(0).sort_values(by=["Theme"])
            merged.rename(columns={f"{col}_x": f"{col} (Reduced)",
                                   f"{col}_y": f"{col}",
                                   "Weight_x": "Weight (Reduced)",
                                   "Weight_y": "Weight"}, inplace=True)

            axes.bar(positions - 0.2, merged[f"{col} (Reduced)"], width=0.4, color=[colors[theme] for theme in themes])
            axes.bar(positions + 0.2, merged[f"{col}"], width=0.4, color=[colors[theme] for theme in themes], edgecolor="white",
                     hatch="///")
            legend_elems.append(Patch(facecolor="white", edgecolor="black", label=f"Full {col}", hatch="///"))

            top = max(top, other_wvs[genre][col].max())
            bottom = min(bottom, other_wvs[genre][col].min())

            plt.xticks(rotation=90)

            ymin = -1
            ymajor = 40 if (top > 0.1 or abs(bottom) > 0.1) else 80 if top > 0.05 else 160
            offset = 0.01 if top > 0.05 else 0.002

        else:
            legend_elems = [Patch(facecolor="white", edgecolor="black", label=col)]
            # if "Rank" not in col and map_to_rosect_wvs[genre][col].max() != 1.0:
            #     map_to_rosect_wvs[genre][col] = scaler.fit_transform(map_to_rosect_wvs[genre][[col]])
            if len(map_to_rosect_wvs[genre]) == 1:
                map_to_rosect_wvs[genre][col] = 1.0
            map_to_rosect_wvs[genre].plot.bar(x="Theme", y=col, ax=axes, rot=90, width=0.5, color=[colors[theme] for theme in themes])
            if col != "Weight":
                ymin = 0
            else:
                ymin = -1
            ymajor = 20
            offset = 0.0001

        axes.set_title(genre, fontsize=32)
        params_dict = {"ymajor": ymajor, "yminor": ymajor * 5, "x_ticklabels": themes, "grid": "-"}
        if len(map_to_rosect_wvs[genre]) > 1:
            params_dict.update({"bottom": bottom - offset, "top": top + offset})
        setup_axis(axes, ymin=ymin, ylabel=col, params_dict=params_dict)

        # axes.grid(axis="x", linestyle="--")

        axes.legend(handles=legend_elems, loc="upper right", fontsize=18)
        plt.margins(x=0.025)
        plt.show()


def plot_theme_weights_by_genre(wvs_df: pd.DataFrame, col: str, genre_list: List = NO_HORROR):
    markers = setup_markers(genre_list)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))
    wvs_df.sort_values(by=["Theme"], inplace=True)
    themes = list(wvs_df["Theme"].unique())
    themes.sort()

    for genre, m in zip(genre_list, markers):
        wvs_df.loc[wvs_df["Genre"] == genre][["Theme", col]].plot(x="Theme", y=col, ax=axes, rot=90,
                                                                  marker=m if len(genre_list) <= len(NO_HORROR) else "",
                                                                  markersize=20, markeredgewidth=2, fillstyle="none", linewidth=2)

    axes.set_title("Theme Weight Rankings by Genre", fontsize=32)
    setup_axis(axes, ymax=22, ymajor=22, yminor=44, x_ticklabels=themes, ylabel=col, bottom=-0.5, top=22.5, grid="-")

    axes.grid(axis="x", linestyle="--")

    axes.legend(genre_list, bbox_to_anchor=(1.005, 1), loc="upper left", fontsize=19)
    plt.margins(x=0.025)
    plt.show()


def get_theme_diffs(wvs1: Dict, wvs2: Dict, genre_list: List = NO_HORROR):
    theme_diffs = {}
    for genre in genre_list:
        g_diff = pd.merge(wvs1[genre], wvs2[genre], on=["Genre", "Theme"], how="outer").fillna(0)
        g_diff.rename(columns={"Frequency Difference_x": "Frequency Difference (Reduced)",
                               "Frequency Difference_y": "Frequency Difference",
                               "Weight_x": "Weight (Reduced)",
                               "Weight_y": "Weight"}, inplace=True)

        error = distance(g_diff[["Frequency Difference (Reduced)", "Weight (Reduced)"]], g_diff[["Frequency Difference", "Weight"]])
        theme_diffs[genre] = pd.DataFrame({"Genre": g_diff["Genre"], "Theme": g_diff["Theme"], "Error": error}).sort_values(by=["Theme"])
        theme_diffs[genre] = theme_diffs[genre].append({"Genre": "Average", "Theme": "Average", "Error": theme_diffs[genre]["Error"].mean()},
                                                       ignore_index=True)

    return theme_diffs


def distance(df1: Union[pd.DataFrame, pd.Series], df2: Union[pd.DataFrame, pd.Series]):
    return np.linalg.norm(df1.values - df2.values, axis=1)


def plot_theme_diffs(theme_diffs_: Dict, colors: Dict, genre_list: List = NO_HORROR):
    for genre in genre_list:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

        theme_diffs = theme_diffs_[genre][theme_diffs_[genre]["Genre"] != "Average"].copy()
        themes = list(theme_diffs["Theme"])

        theme_diffs.plot.bar(x="Theme", y="Error", ax=axes, width=0.5, rot=90, color=[colors[theme] for theme in themes])

        axes.set_title(genre, fontsize=32)
        ymajor = 10 if theme_diffs["Error"].max() > 0.7 else 20
        setup_axis(axes, ymajor=ymajor, yminor=ymajor * 4,
                   x_ticklabels=themes, ylabel="No Reduction Theme Error",
                   top=theme_diffs["Error"].max() + 0.02,
                   grid="-", minor_grid=":")

        axes.grid(axis="x", linestyle="none")
        axes.get_legend().remove()

        plt.show()


def plot_avg_freq_diff_by_genre(map_to_rosect_wvs: Dict, colors: Dict, other_wvs: Optional[Dict] = None, genre_list: List = NO_HORROR, **kwargs):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(30, 15))

    avg_freq_diffs = pd.DataFrame(
        [{"Genre": genre, "Average Frequency Difference": map_to_rosect_wvs[genre]["Frequency Difference"].abs().mean()} for genre in
         genre_list])

    legend_elems = [Patch(facecolor="white", edgecolor="black", label="Reduced Average Frequency Difference")]

    top = avg_freq_diffs["Average Frequency Difference"].max()
    bottom = avg_freq_diffs["Average Frequency Difference"].min()

    if other_wvs is not None:
        other_avg_freq_diffs = pd.DataFrame(
            [{"Genre": genre, "Average Frequency Difference": other_wvs[genre]["Frequency Difference"].abs().mean()} for genre in genre_list])
        positions = np.arange(0, len(genre_list))

        merged = pd.merge(avg_freq_diffs, other_avg_freq_diffs, on="Genre", how="outer").fillna(0)
        merged.rename(columns={"Average Frequency Difference_x": "Average Frequency Difference (Reduced)",
                               "Average Frequency Difference_y": "Average Frequency Difference"}, inplace=True)

        axes.bar(positions - 0.2, merged["Average Frequency Difference (Reduced)"], width=0.4, color=[colors[genre] for genre in genre_list])
        axes.bar(positions + 0.2, merged["Average Frequency Difference"], width=0.4, color=[colors[genre] for genre in genre_list],
                 edgecolor="white", hatch="///")
        legend_elems.append(Patch(facecolor="white", edgecolor="black", label="Full Average Frequency Difference", hatch="///"))

        top = max(top, other_avg_freq_diffs["Average Frequency Difference"].max())
        bottom = min(bottom, other_avg_freq_diffs["Average Frequency Difference"].min())

        plt.xticks(rotation=90)
        display_df(merged.append({"Genre": "Average",
                                  "Average Frequency Difference (Reduced)": merged["Average Frequency Difference (Reduced)"].mean(),
                                  "Average Frequency Difference": merged["Average Frequency Difference"].mean()}, ignore_index=True))

    else:
        avg_freq_diffs.plot.bar(x="Genre", y="Average Frequency Difference", ax=axes, rot=90, width=0.5,
                                color=[colors[genre] for genre in genre_list])
        display_df(
            avg_freq_diffs.append({"Genre": "Average", "Average Frequency Difference": avg_freq_diffs["Average Frequency Difference"].mean()},
                                  ignore_index=True))

    ymajor = 40 if (top > 0.1 or abs(bottom) > 0.1) else 80 if top > 0.05 else 160
    offset = 0.01 if top > 0.05 else 0.002
    setup_axis(axes, ymin=-1, ymajor=ymajor, yminor=ymajor * 5,
               x_ticklabels=genre_list,
               ylabel="Magnitude of Avg Frequency Difference",
               bottom=bottom - offset,
               top=top + offset,
               grid="-", minor_grid=":")

    axes.set_title(kwargs.get("title", ""), fontsize=32)
    axes.grid(axis="x", linestyle="--")

    axes.legend(handles=legend_elems, loc="upper right", fontsize=18)
    plt.margins(x=0.025)
    plt.show()
