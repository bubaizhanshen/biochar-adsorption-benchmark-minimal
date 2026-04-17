from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "shap_cross_model_ec"
OUT_DIR = ROOT / "outputs" / "figure6_cbz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_NAME = "CBZ"

BP_FEATURES = {
    "C",
    "H",
    "O",
    "N",
    "(O+N)/C",
    "Ash",
    "H/C",
    "O/C",
    "N/C",
    "Surface area",
    "Pore volume",
    "Average pore size",
}
PC_FEATURES = {"Pyrolysis temperature", "Pyrolysis time"}
AC_FEATURES = {
    "Adsorption time",
    "Initial concentration",
    "Solution pH",
    "RPM",
    "Volume",
    "Adsorbent dosage",
    "Adsorption temperature",
    "Ion concentration",
    "Humic acid",
}

CAT_COLORS = {
    "BP": "#536878",
    "PC": "#D07A28",
    "AC": "#2E8B83",
    "Other": "#A8ADB3",
}


def configure_arial_font() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]


def darken(color: str, factor: float) -> tuple[float, float, float]:
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(rgb * factor, 0, 1))


def categorize(feature: str) -> str:
    if feature in BP_FEATURES:
        return "BP"
    if feature in PC_FEATURES:
        return "PC"
    if feature in AC_FEATURES:
        return "AC"
    return "Other"


def load_cbz_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_df = pd.read_csv(BASE_DIR / "feature_level_summary.csv")
    task_df = pd.read_csv(BASE_DIR / "task_summary.csv")
    feature_df = feature_df[feature_df["task"] == TASK_NAME].copy()
    task_df = task_df[task_df["task"] == TASK_NAME].copy()
    if feature_df.empty or task_df.empty:
        raise RuntimeError("CBZ results were not found in the cross-model SHAP output directory.")
    return feature_df, task_df


def aggregate_importance(feature_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        feature_df.groupby("feature", as_index=False)[["rf_importance", "xgb_importance", "rf_rank", "xgb_rank"]]
        .mean()
        .sort_values("feature")
        .reset_index(drop=True)
    )
    agg["category"] = agg["feature"].map(categorize)
    agg["xgb_pct"] = agg["xgb_importance"] / agg["xgb_importance"].sum() * 100
    agg["rf_pct"] = agg["rf_importance"] / agg["rf_importance"].sum() * 100
    agg["combined_pct"] = (agg["xgb_pct"] + agg["rf_pct"]) / 2
    return agg


def category_share_table(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, value_col in [
        ("XGB", "xgb_pct"),
        ("RF", "rf_pct"),
        ("Combined", "combined_pct"),
    ]:
        sub = agg.groupby("category", as_index=False)[value_col].sum()
        sub["model"] = model_name
        sub["share_pct"] = sub[value_col]
        rows.append(sub[["model", "category", "share_pct"]])
    out = pd.concat(rows, ignore_index=True)
    order = pd.Categorical(out["category"], categories=["BP", "PC", "AC", "Other"], ordered=True)
    return out.assign(category=order).sort_values(["model", "category"]).reset_index(drop=True)


def plot_bar_panel(ax, sub: pd.DataFrame, value_col: str, title: str, value_fmt: str) -> None:
    sub = sub.sort_values(value_col, ascending=True).copy()
    y = np.arange(len(sub))
    bar_colors = [CAT_COLORS.get(cat, CAT_COLORS["Other"]) for cat in sub["category"]]
    edge_colors = [darken(color, 0.72) for color in bar_colors]
    bars = ax.barh(y, sub[value_col], color=bar_colors, alpha=0.96, edgecolor=edge_colors, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["feature"], fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, value in zip(bars, sub[value_col]):
        ax.text(
            bar.get_width() + max(sub[value_col]) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            value_fmt.format(value),
            va="center",
            ha="left",
            fontsize=8.5,
        )


def plot_pseudo_3d_pie(ax, labels: list[str], values: list[float], title: str, show_legend: bool = False) -> None:
    depth = 18
    radius = 1.0
    top_colors = [CAT_COLORS[label] for label in labels]
    startangle = 110

    for layer in range(depth, 0, -1):
        factor = 0.62 + 0.015 * layer
        layer_colors = [darken(color, factor) for color in top_colors]
        ax.pie(
            values,
            colors=layer_colors,
            startangle=startangle,
            radius=radius,
            center=(0, -0.018 * layer),
            wedgeprops={"linewidth": 0.4, "edgecolor": darken("#333333", 0.9)},
        )

    wedges, _, autotexts = ax.pie(
        values,
        colors=top_colors,
        startangle=startangle,
        radius=radius,
        center=(0, 0),
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.75,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        shadow=False,
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")

    if show_legend:
        ax.legend(
            wedges,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False,
            fontsize=9,
        )
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.55, 1.15)


def add_metrics_box(ax, summary_row: pd.Series) -> None:
    text_lines = [
        "Shared RS split, same full feature set",
        f"RF mean test R2: {summary_row['rf_r2_mean']:.4f}",
        f"XGB mean test R2: {summary_row['xgb_r2_mean']:.4f}",
        f"Prediction corr.: {summary_row['prediction_corr_mean']:.4f}",
        f"SHAP-rank Spearman: {summary_row['shap_rank_spearman_mean']:.4f}",
        f"Top-5 overlap: {summary_row['top_k_overlap_mean']:.4f}",
        f"Top-1 match rate: {summary_row['top1_match_rate']:.4f}",
    ]
    box = FancyBboxPatch(
        (0.04, 0.05),
        0.92,
        0.23,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=0.9,
        edgecolor="#999999",
        facecolor="#f6f7f8",
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        0.07,
        0.255,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.4,
        linespacing=1.4,
    )


def annotate_panel(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def save_tables(agg: pd.DataFrame, cat_df: pd.DataFrame) -> None:
    xgb_table = agg.sort_values("xgb_importance", ascending=False).copy()
    rf_table = agg.sort_values("rf_importance", ascending=False).copy()
    combined_table = agg.sort_values("combined_pct", ascending=False).copy()

    out_xlsx = OUT_DIR / "cbz_cross_model_importance.xlsx"
    with pd.ExcelWriter(out_xlsx) as writer:
        xgb_table.to_excel(writer, sheet_name="xgb_ranking", index=False)
        rf_table.to_excel(writer, sheet_name="rf_ranking", index=False)
        combined_table.to_excel(writer, sheet_name="combined_ranking", index=False)
        cat_df.to_excel(writer, sheet_name="category_share", index=False)

    xgb_table.to_csv(OUT_DIR / "cbz_xgb_ranking.csv", index=False)
    rf_table.to_csv(OUT_DIR / "cbz_rf_ranking.csv", index=False)
    combined_table.to_csv(OUT_DIR / "cbz_combined_ranking.csv", index=False)
    cat_df.to_csv(OUT_DIR / "cbz_category_share.csv", index=False)


def main() -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    configure_arial_font()

    feature_df, task_df = load_cbz_tables()
    agg = aggregate_importance(feature_df)
    cat_df = category_share_table(agg)
    summary_row = task_df.iloc[0]
    save_tables(agg, cat_df)

    xgb_top = agg.sort_values("xgb_importance", ascending=False).head(10)
    rf_top = agg.sort_values("rf_importance", ascending=False).head(10)
    combined_top = agg.sort_values("combined_pct", ascending=False).head(10)

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    gs = fig.add_gridspec(1, 3, left=0.045, right=0.985, top=0.84, bottom=0.11, wspace=0.34)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    plot_bar_panel(ax_a, xgb_top, "xgb_importance", "XGB SHAP Importance", "{:.3f}")
    ax_a.set_xlabel("Mean |SHAP value|")
    annotate_panel(ax_a, "A")

    plot_bar_panel(ax_b, rf_top, "rf_importance", "RF SHAP Importance", "{:.3f}")
    ax_b.set_xlabel("Mean |SHAP value|")
    annotate_panel(ax_b, "B")

    plot_bar_panel(ax_c, combined_top, "combined_pct", "Combined Ranking", "{:.2f}%")
    ax_c.set_xlabel("Mean normalized contribution")
    annotate_panel(ax_c, "C")
    add_metrics_box(ax_c, summary_row)

    for ax, model in zip([ax_a, ax_b, ax_c], ["XGB", "RF", "Combined"]):
        sub = cat_df[cat_df["model"] == model].copy()
        labels = sub["category"].astype(str).tolist()
        values = sub["share_pct"].tolist()
        inset = ax.inset_axes([0.55, 0.49, 0.40, 0.38])
        plot_pseudo_3d_pie(inset, labels, values, f"{model} share")

    fig.suptitle(
        "CBZ Cross-Model Feature Attribution Under Random Splitting\nDataset III (EC), full feature set, shared RS splits",
        fontsize=16,
        fontweight="bold",
        y=0.965,
    )
    legend_handles = [
        Patch(facecolor=CAT_COLORS["BP"], edgecolor=darken(CAT_COLORS["BP"], 0.72), label="BP: biochar properties"),
        Patch(facecolor=CAT_COLORS["PC"], edgecolor=darken(CAT_COLORS["PC"], 0.72), label="PC: pyrolysis conditions"),
        Patch(facecolor=CAT_COLORS["AC"], edgecolor=darken(CAT_COLORS["AC"], 0.72), label="AC: adsorption conditions"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=3,
        frameon=False,
        fontsize=10,
        handlelength=1.5,
        columnspacing=1.8,
    )
    fig.text(
        0.5,
        0.03,
        "Bar colors and pies follow the same BP-PC-AC coding. Pies show category-level share of mean absolute SHAP importance.",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    for ext in ["svg", "pdf", "png"]:
        fig.savefig(OUT_DIR / f"cbz_cross_model_panels_a4.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
