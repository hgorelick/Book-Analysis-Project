__all__ = ['display_df', 'setup_axis', 'process_and_scale', 'create_cmap', 'get_display_df', 'setup_markers',
           'finish_plot', 'setup_search_plot', 'filter_out_zeros']


from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from IPython.display import display, HTML
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from tqdm.notebook import tqdm_notebook as tqdm

from notebook_utils.constants import scaler, NO_HORROR

tqdm.pandas()


def display_df(df: pd.DataFrame, header: Optional[str] = None, max_rows: Optional[int] = None,
               add_break: bool = False, formatters: Optional[Dict] = None, index: bool = False):
    html = ""
    if header is not None:
        html += header
        
    if format is not None:
        html += df.to_html(index=index, max_rows=max_rows, formatters=formatters)
    else:
        html += df.to_html(index=index, max_rows=max_rows)
        
    if add_break:
        html += "<br>"
    display(HTML(html))
    

def setup_axis(ax, xmin: Optional[Union[int, float]] = 0, xmax: Optional[Union[int, float]] = 1,
               ymin: Union[int, float] = 0, ymax: Union[int, float] = 1,
               xlabel: str = "", ylabel: str = "", **kwargs):
    """
    - xmajor: int = 20
    - xminor: int = 100
    - ymajor: int = 20
    - yminor: int = 100
    - x_ticklabels: List = None
    - x_ticklabel_size: Union[int, float] = 20
    - y_ticklabel_size: Union[int, float] = 24
    - xlabel_size: Union[int, float] = 28
    - xlabel_pad: Union[int, float] = 20
    - ylabel_size: Union[int, float] = 32
    - ylabel_pad: Union[int, float] = 30
    - left: Union[int, float] = None
    - right: Union[int, float] = None
    - bottom: Union[int, float] = None
    - top: Union[int, float] = None
    - grid: str = "--"
    - minor_grid: str = None
    """

    if kwargs.get("params_dict", None) is not None:
        kwargs = kwargs["params_dict"]

    if "x_ticklabels" not in kwargs and xmin is not None:
        if xmin < 0:
            xmajor = kwargs.get("xmajor", xmax * 4)
            ax.set_xticks(np.arange(xmin if xmajor >= 40 else 0, xmax + 1, xmax / xmajor))
            
            xminor = kwargs.get("xminor", xmajor * 10)
            ax.set_xticks(np.arange(xmin, xmax + 1, xmax / xminor), minor=True)
        else:
            if xmax == 349:
                major = np.linspace(2, 350, 13)
                minor = np.linspace(2, 350, 25)
            else:
                major = np.linspace(xmin, xmax, kwargs.get("xmajor", 20) + (1 if (xmin != 23 and xmin != 22) else 0))
                minor = np.linspace(xmin, xmax, kwargs.get("xminor", 100) + (1 if (xmin != 23 and xmin != 22) else -1))
            if kwargs.get("reverse", False):
                major = major[::-1]
                minor = minor[::-1]
            print(major)
            ax.set_xticks(major)
            ax.set_xticks(minor, minor=True)
    elif "x_ticklabels" in kwargs:
        ax.set_xticks(np.arange(xmin, len(kwargs["x_ticklabels"])))
        ax.set_xticklabels(kwargs["x_ticklabels"])
    
    ax.set_yticks(np.linspace(ymin, ymax, kwargs.get("ymajor", 20) + 1))
    ax.set_yticks(np.linspace(ymin, ymax, kwargs.get("yminor", 100) + 1), minor=True)
    
    ax.tick_params(axis="x", labelsize=kwargs.get("x_ticklabel_size", 20))
    ax.tick_params(axis="y", labelsize=kwargs.get("y_ticklabel_size", 24))
    
    xlabel_size = kwargs.get("xlabel_size", 0 if xlabel == "" else 28)
    xlabel_pad = kwargs.get("xlabel_pad", None if xlabel == "" else 20)
    ax.set_xlabel(xlabel, fontsize=xlabel_size, labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, fontsize=kwargs.get("ylabel_size", 32), labelpad=kwargs.get("ylabel_pad", 30))
    
    if "left" in kwargs:
        ax.set_xlim(left=kwargs["left"])
    if "right" in kwargs:
        ax.set_xlim(right=kwargs["right"])
    if "bottom" in kwargs:
        ax.set_ylim(bottom=kwargs["bottom"])
    if "top" in kwargs:
        ax.set_ylim(top=kwargs["top"])
    
    ax.grid(linestyle=kwargs.get("grid", "--"))
    ax.grid(linestyle=kwargs.get("minor_grid", "none"), which="minor")


def process_and_scale(data: Union[List, pd.DataFrame], model: str, n_cols: int = 5):
    if isinstance(data, list):
        data_df = pd.concat(data).fillna(0).reset_index(drop=True)
    else:
        data_df = data.reset_index(drop=True)

    data_df = data_df.loc[:, ~data_df.columns.duplicated()]
    nominal = data_df[["Book #", "@Genre", "@Outcome", "@Downloads"]].reset_index(drop=True)
    data_df.drop(columns=["Book #", "@Genre", "@Outcome", "@Downloads"], inplace=True)
    
    if model == "phrasal" or model == "clausal":
        data_df = data_df / data_df.sum()

    # if "lex" not in model:
    data_df_scaled = scaler.fit_transform(data_df)
    data_df_scaled = pd.DataFrame(data_df_scaled, columns=data_df.columns)
    data_df_scaled.insert(0, "@Genre", nominal["@Genre"])
    data_df_scaled.insert(0, "Book #", nominal["Book #"])
    data_df_scaled["@Outcome"] = nominal["@Outcome"]
    data_df_scaled["@Downloads"] = nominal["@Downloads"]

    if len(data_df_scaled.columns) > n_cols + 3:
        to_display = get_display_df(data_df_scaled, n_cols)
    else:
        to_display = data_df_scaled.copy()
    
    return data_df_scaled, to_display


def get_display_df(df: pd.DataFrame, n_cols: int = 5):
    to_display = df.iloc[:, :n_cols].copy()
    to_display["..."] = "..."
    to_display = pd.concat([to_display, df.iloc[:, -n_cols:]], axis=1)
    return to_display


def create_cmap(cmap, items: List, as_dict: bool = True):
    color_list = [to_hex(c) for c in cycler("color", cmap(np.linspace(0, 1, len(items)))).by_key()["color"]]
    return dict(zip(items, color_list)) if as_dict else color_list


def setup_markers(genre_list: List[str] = NO_HORROR):
    markers = ["o", "P", "s", "D", "p", "v", "H", "*"]
    i = 0
    while len(genre_list) > len(markers):
        markers.append(markers[i])
        i += 1
        if i >= len(markers):
            i = 0
    return markers


def finish_plot(axes, df: pd.DataFrame, max_steps: int, tuned_params: List, x: str, all: bool = False, for_paper: bool = False, **kwargs):
    tuned_params_df = pd.DataFrame(tuned_params)
    if all:
        display_df(df)
    else:
        to_append = {}
        for col, data in tuned_params_df.items():
            if col == "Genre":
                to_append[col] = "Average"
            else:
                to_append[col] = tuned_params_df[col].mean()
        display_df(tuned_params_df.append(to_append, ignore_index=True))

    min_x = df[x].min()
    max_x = df[x].max()
    min_acc = df["Accuracy"].min()
    max_acc = df["Accuracy"].max()

    if max_steps >= 35:
        xmajor = max_steps // 100000 if max_steps > 50000 else max_steps // 3500 if max_steps > 20000 \
            else max_steps // 500 if max_steps > 5000 else max_steps // 100 if max_steps >= 1000 \
            else 10 if max_steps > 250 else \
            max_steps // 10 if max_steps > 100 else max_steps // 5 if max_steps > 35 else max_steps
        xminor = xmajor * 5 if max_steps < 300 else xmajor * 4  # if max_steps != 349 else (xmajor - 1) * 4
        offset_x = 5000 if max_steps > 50000 else 400 if max_steps > 20000 \
            else 100 if max_steps > 5000 else 10 if max_steps >= 1000 else 5 if max_steps > 100 \
            else 1 if max_steps > 35 else 0.25
        right = min_x - 5000 if max_steps > 50000 else min_x - (min_x % 5) - offset_x if max_steps >= 1000 else min_x - offset_x
        # print(max_steps, min_x, right)
    else:
        if max_x <= 6:
            xmajor = max_steps * 4
            xminor = xmajor * 5
            offset_x = 0.05
        else:
            xmajor = max_steps * 2 if max_steps <= 15 else max_steps
            xminor = xmajor * 2
            offset_x = 0.25
        right = min_x - offset_x

    if for_paper:
        ymin = 0.5
        ymax = 1.0  # if max_acc >= 0.9 else 0.85 if max_acc >= 0.75 else 0.70
        offset_y = 0.01  # if max_acc >= 0.75 else 0.001
        ymajor = 10
        yminor = 20
    elif (max_acc - min_acc) / 2 < 0.05:
        ymin = np.round(min_acc / 0.0001, 0) * 0.0001
        ymax = np.round(max_acc / 0.0001, 0) * 0.0001
        # if ymin > min_acc:
        #     ymin -= 0.0001
        # if ymax < max_acc:
        #     ymax += 0.0001
        offset_y = 0.00005
        ymajor = 10
        yminor = 20
    elif (max_acc - min_acc) / 2 < 0.2:
        ymin = np.round(min_acc / 0.001, 0) * 0.001
        ymax = np.round(max_acc / 0.001, 0) * 0.001
        if ymin > min_acc:
            ymin -= 0.001
        if ymax < max_acc:
            ymax += 0.001
        offset_y = 0.015
        ymajor = 10
        yminor = 20
    else:
        ymin = 0
        ymax = 1
        offset_y = 0.025
        ymajor = 20
        yminor = 100

    if for_paper:
        x_ticklabel_size = 14
        y_ticklabel_size = 14
        xlabel_size = 16
        ylabel_size = 16
        xlabel_pad = 12
        ylabel_pad = 18
    else:
        x_ticklabel_size = 20
        y_ticklabel_size = 24
        xlabel_size = 28
        ylabel_size = 32
        xlabel_pad = 20
        ylabel_pad = 30

    if not kwargs.get("reverse"):
        left = right
        right = max_x + offset_x
    else:
        left = max_x + offset_x
    print(min_x if not kwargs.get("reverse", False) else max_x, max_x if not kwargs.get("reverse", False) else min_x)
    setup_axis(axes, xmin=min_x if not kwargs.get("reverse", False) else max_x, xmax=max_x if not kwargs.get("reverse", False) else min_x,
               ymin=ymin, ymax=ymax,
               xmajor=xmajor, xminor=xminor,
               ymajor=ymajor, yminor=yminor,
               xlabel=x if x != "Margin Width" else "Difference of $\\upsilon^{-}$ and $\\upsilon^{+}$", ylabel="Accuracy",
               xlabel_size=xlabel_size,
               ylabel_size=ylabel_size,
               xlabel_pad=xlabel_pad,
               ylabel_pad=ylabel_pad,
               x_ticklabel_size=x_ticklabel_size,
               y_ticklabel_size=y_ticklabel_size,
               left=left, right=right,
               bottom=ymin - offset_y, top=ymax + offset_y,
               grid=":", minor_grid=":", reverse=kwargs.get("reverse", False))

    # axes.legend(genre_list, bbox_to_anchor=(0, 1.05), loc="upper left", fontsize=14, ncol=len(genre_list))
    return tuned_params_df


def setup_search_plot(df: pd.DataFrame, x: str, width: int = 30, height: int = 17):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(width, height))
    exh_df = df.copy().sort_values(by=[x]).reset_index(drop=True)
    if exh_df[x].max() > 10:
        max_steps = exh_df[x].max()
    else:
        max_steps = 10
    return fig, axes, exh_df, max_steps


def filter_out_zeros(df: pd.DataFrame, cols: Optional[List] = None):
    # if cols is not None:
    #     to_drop = [i for i in tqdm(range(len(cols)), postfix="-- REMOVING ZEROS FROM UNIGRAM") if all(df[:, i] == 0)]
    #     keep = list(set(list(i for i in range(len(cols)))).difference(set(to_drop)))
    #     new_cols = [cols[i] for i in keep]
    #     return pd.DataFrame(df[keep], columns=new_cols)
    try:
        return df.loc[:, (df != 0).any(axis=0)]
    except (MemoryError, AttributeError):
        idx = np.argwhere(np.all(df[..., :] == 0, axis=0))
        df = np.delete(df, idx, axis=1)
        keep = list(set(list(i for i in range(len(cols)))).difference(set(idx)))
        new_cols = [cols[i] for i in keep]
        return pd.DataFrame(df, columns=new_cols, dtype="int32")
        # if cols is not None:
        #     to_drop = [i for i in tqdm(range(len(cols)), postfix="-- REMOVING ZEROS FROM UNIGRAM") if all(df[:, i] == 0)]
        #     keep = list(set(list(i for i in range(len(cols)))).difference(set(to_drop)))
        #     new_cols = [cols[i] for i in keep]
        #     return pd.DataFrame(df[:, keep], columns=new_cols)
