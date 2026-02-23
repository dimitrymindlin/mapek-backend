"""Lightweight statistical plotting helpers used by the unified analysis pipeline."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


def _to_series(data: Iterable, name: str) -> pd.Series:
    series = pd.Series(data, name=name)
    return pd.to_numeric(series, errors="coerce")


def get_groups_two_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    x_label: str,
    y_label: str,
) -> Tuple[pd.Series, pd.Series, str, str]:
    """Return the value series and labels for the provided dataframes."""
    label1 = _first_label(df1, x_label, fallback="group_1")
    label2 = _first_label(df2, x_label, fallback="group_2")

    group1 = _to_series(df1[y_label], name=f"{label1}_{y_label}")
    group2 = _to_series(df2[y_label], name=f"{label2}_{y_label}")

    return group1, group2, label1, label2


def get_groups_one_df(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    groups_to_compare: Optional[Sequence[str]] = None,
) -> Tuple[pd.Series, pd.Series, str, str]:
    """Return the value series for two groups residing in a single dataframe."""
    if groups_to_compare:
        labels = list(groups_to_compare)
    else:
        labels = [lbl for lbl in df[x_label].dropna().unique()[:2]]

    if len(labels) < 2:
        raise ValueError("Need at least two groups to compare")

    group1 = _to_series(df[df[x_label] == labels[0]][y_label], name=f"{labels[0]}_{y_label}")
    group2 = _to_series(df[df[x_label] == labels[1]][y_label], name=f"{labels[1]}_{y_label}")

    return group1, group2, str(labels[0]), str(labels[1])


def _first_label(df: pd.DataFrame, column: str, fallback: str) -> str:
    values = df[column].dropna().astype(str).unique()
    return values[0] if len(values) else fallback


def _is_normal(series: pd.Series) -> bool:
    cleaned = series.dropna()
    if len(cleaned) < 3:
        return False

    try:
        stat, p_value = stats.shapiro(cleaned)
    except Exception:
        return False
    return p_value >= 0.05


def is_t_test_applicable(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    df2: Optional[pd.DataFrame] = None,
) -> bool:
    """Best-effort replica of the original helper to gate t-test usage."""
    if df2 is None:
        group1, group2, *_ = get_groups_one_df(df, x_label, y_label)
    else:
        group1, group2, *_ = get_groups_two_dfs(df, df2, x_label, y_label)

    group1 = group1.dropna()
    group2 = group2.dropna()

    if len(group1) < 3 or len(group2) < 3:
        return False

    normal1 = _is_normal(group1)
    normal2 = _is_normal(group2)

    if not (normal1 and normal2):
        return False

    try:
        _, p_value = stats.levene(group1, group2)
    except Exception:
        p_value = 0.0

    return p_value >= 0.05


def plot_box_with_significance_bars(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    ttest: bool = False,
    save: bool = True,
    ax: Optional[plt.Axes] = None,
    y_label_name: Optional[str] = None,
    df2: Optional[pd.DataFrame] = None,
    groups_to_compare: Optional[Sequence[str]] = None,
) -> None:
    """Plot a simple two-group boxplot with an optional significance bar."""
    if df2 is None:
        group1, group2, label1, label2 = get_groups_one_df(df, x_label, y_label, groups_to_compare)
        combined = df.copy()
    else:
        group1, group2, label1, label2 = get_groups_two_dfs(df, df2, x_label, y_label)
        combined = pd.concat([df, df2], ignore_index=True)

    group1 = group1.dropna()
    group2 = group2.dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    categories = [str(label1), str(label2)]
    combined = combined.copy()
    combined[x_label] = combined[x_label].astype(str)

    box_data = [combined.loc[combined[x_label] == cat, y_label].dropna().to_numpy() for cat in categories]

    positions = np.arange(1, len(categories) + 1)
    bp = ax.boxplot(box_data, positions=positions, labels=categories, widths=0.55, patch_artist=True)
    for patch, color in zip(bp['boxes'], ["#223f9c", "#f28e2c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    if y_label_name:
        ax.set_ylabel(y_label_name)
    ax.set_title(title)

    stat, p_value = _compute_p_value(group1, group2, ttest)

    if p_value is not None and p_value < 0.05:
        y_max = max(np.max(vals) if len(vals) else 0 for vals in box_data)
        y_min = min(np.min(vals) if len(vals) else 0 for vals in box_data)
        spread = y_max - y_min if y_max != y_min else 1.0
        bar_height = y_max + 0.08 * spread
        x1, x2 = positions[0], positions[1]
        ax.plot([x1, x1, x2, x2], [bar_height, bar_height + 0.03 * spread, bar_height + 0.03 * spread, bar_height], color="k", linewidth=1.2)

        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        ax.text((x1 + x2) / 2, bar_height + 0.05 * spread, stars, ha="center", va="bottom", fontsize=11)

    if save and ax is not None and ax.figure is not None:
        safe_title = title.replace(" ", "_").lower()
        ax.figure.savefig(f"{safe_title}.png", bbox_inches="tight", dpi=150)


def _compute_p_value(group1: pd.Series, group2: pd.Series, ttest: bool) -> Tuple[Optional[float], Optional[float]]:
    if len(group1) == 0 or len(group2) == 0:
        return None, None

    if ttest:
        stat, p_value = stats.ttest_ind(group1, group2, equal_var=False, nan_policy="omit")
    else:
        stat, p_value = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    return stat, p_value


__all__ = [
    "get_groups_one_df",
    "get_groups_two_dfs",
    "is_t_test_applicable",
    "plot_box_with_significance_bars",
]
