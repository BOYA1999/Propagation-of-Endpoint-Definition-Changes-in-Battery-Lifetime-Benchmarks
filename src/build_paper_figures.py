import argparse
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


plt.style.use(Path(__file__).with_name("academic.mplstyle"))
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "endpoint-definition-audit"

OUT = None
DATA = None

NAVY = "#335F7D"
BLUE = "#5B82A6"
TEAL = "#4E948D"
ORANGE = "#D18443"
PURPLE = "#786AA3"
RED = "#B95850"
GREY = "#AEB7BD"
MID = "#6E7B83"
DARK = "#25343D"
LIGHT = "#F3F6F7"
GRID = "#E2E7E9"


def style_axis(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.8, width=0.7, color=MID)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)


def panel(ax, label, x=-0.08, y=1.02):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=DARK, ha="left", va="bottom")


def save_figure(fig, number, conclusion):
    base = OUT / f"Fig{number}"
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".eps"), metadata={"Creator": "Matplotlib"})
    fig.savefig(base.with_suffix(".png"), dpi=600, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    eps_path = base.with_suffix(".eps")
    eps_text = eps_path.read_text(encoding="utf-8")
    eps_path.write_text(re.sub(r"^%%CreationDate:.*$", "%%CreationDate: Thu Jan 01 00:00:00 1970", eps_text, flags=re.MULTILINE), encoding="utf-8")
    (OUT / f"Fig{number}_conclusion.txt").write_text(conclusion + "\n", encoding="utf-8")


def figure1():
    fig = plt.figure(figsize=(6.8, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.18, 2.15])

    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.015, 0.12), 0.97, 0.72, boxstyle="round,pad=0.012,rounding_size=0.02", transform=ax.transAxes, facecolor=LIGHT, edgecolor="#CED7DC", linewidth=0.8))
    ax.text(0.045, 0.73, "Executable endpoint contract", transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=DARK, va="center")
    components = [
        ("Capacity", "reference"),
        ("Observation", "schedule"),
        ("Time", "coordinate"),
        ("Crossing", "rule"),
        ("Code", "revision"),
    ]
    starts = np.linspace(0.055, 0.78, 5)
    for index, ((head, sub), x0) in enumerate(zip(components, starts)):
        edge = ORANGE if index == 4 else BLUE
        face = "#FFF7F0" if index == 4 else "white"
        ax.add_patch(FancyBboxPatch((x0, 0.27), 0.15, 0.3, boxstyle="round,pad=0.01,rounding_size=0.014", transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.95))
        ax.text(x0 + 0.075, 0.445, head, transform=ax.transAxes, ha="center", va="center", fontsize=7.8, fontweight="bold", color=DARK)
        ax.text(x0 + 0.075, 0.34, sub, transform=ax.transAxes, ha="center", va="center", fontsize=7.3, color=MID)
        if index < 4:
            ax.text(x0 + 0.165, 0.42, "+", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=MID)
    panel(ax, "a", x=-0.015, y=0.9)

    ax = fig.add_subplot(gs[1])
    ax.axis("off")
    ax.add_patch(FancyArrowPatch((0.5, 0.98), (0.5, 0.02), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=10, linewidth=0.85, color=MID))

    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.045, 0.81), 0.91, 0.12, boxstyle="round,pad=0.008,rounding_size=0.012", transform=ax.transAxes, facecolor=NAVY, edgecolor=NAVY, linewidth=0.8))
    ax.text(0.5, 0.87, "Fixed cells · split · early-cycle inputs · model families · seeds", transform=ax.transAxes, color="white", fontsize=8.1, fontweight="bold", ha="center", va="center")
    cards = [
        (0.035, 0.46, 0.27, ORANGE, "Target geometry", "252 / 823 differ", "two source regimes"),
        (0.33, 0.46, 0.195, BLUE, "Evaluation", "44 labels changed", "114 direct effects = 0"),
        (0.55, 0.46, 0.195, TEAL, "Fitting", "predictions move", "49–75% on unchanged"),
        (0.77, 0.46, 0.195, PURPLE, "Intervals", "metric dependent", "one-cell transitions"),
    ]
    for x0, y0, width, color, title, main, sub in cards:
        ax.add_patch(FancyArrowPatch((x0 + width / 2, 0.805), (x0 + width / 2, y0 + 0.26), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=8, linewidth=0.7, color="#98A4AA"))
        ax.add_patch(FancyBboxPatch((x0, y0), width, 0.26, boxstyle="round,pad=0.012,rounding_size=0.015", transform=ax.transAxes, facecolor="white", edgecolor="#CDD5D9", linewidth=0.75))
        ax.add_patch(FancyBboxPatch((x0, y0 + 0.215), width, 0.045, boxstyle="round,pad=0.002,rounding_size=0.01", transform=ax.transAxes, facecolor=color, edgecolor=color, linewidth=0))
        ax.text(x0 + 0.018, y0 + 0.17, title, transform=ax.transAxes, fontsize=7.7, fontweight="bold", color=DARK, va="center")
        ax.text(x0 + 0.018, y0 + 0.105, main, transform=ax.transAxes, fontsize=7.6, color=color, fontweight="bold", va="center")
        ax.text(x0 + 0.018, y0 + 0.05, sub, transform=ax.transAxes, fontsize=7.1, color=MID, va="center")
    ax.text(0.5, 0.21, "Benchmark result", transform=ax.transAxes, fontsize=7.6, fontweight="bold", color=DARK, ha="center")
    ax.text(0.5, 0.12, "target definition  ×  fitted response  ×  evaluation metric  ×  interval calibration", transform=ax.transAxes, fontsize=8.2, color=NAVY, fontweight="bold", ha="center")
    panel(ax, "b", x=-0.015, y=0.96)
    save_figure(fig, 1, "Endpoint execution affects a fixed benchmark through parallel target, evaluation, fitting, and interval pathways.")


def figure2():
    cells = pd.read_csv(DATA / "Fig2_target_geometry_cells.csv")
    cells["labels_differ"] = cells["labels_differ"].astype(str).str.lower().eq("true")
    fig = plt.figure(figsize=(6.8, 5.3), constrained_layout=True)
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1.02, 1.02], hspace=0.38, wspace=0.72)

    ax = fig.add_subplot(gs[:, :3])
    unchanged = cells[~cells.labels_differ]
    ax.scatter(unchanged.release_eol, unchanged.current_generator_eol, s=5.5, color="#D5DADD", linewidths=0, rasterized=True)
    source_specs = [("ISU_ILCC", "#D99B65", "o"), ("Tongji", "#978DB8", "^")]
    for source, color, marker in source_specs:
        data = cells[(cells.dataset_group == source) & cells.labels_differ]
        ax.scatter(data.release_eol, data.current_generator_eol, s=16, color=color, marker=marker, edgecolor="white", linewidth=0.25, label=f"{source.replace('_', '-')}  n={len(data)}", rasterized=True)
    limits = [55, 3700]
    ax.plot(limits, limits, linestyle=(0, (4, 3)), color=DARK, linewidth=0.8)
    outlier = cells.loc[cells.absolute_delta_cycles.idxmax()]
    ax.annotate("G34C2\n3109 → 317", (outlier.release_eol, outlier.current_generator_eol), xytext=(-48, 16), textcoords="offset points", color=RED, fontsize=7.4, fontweight="bold", ha="center", arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.7})
    ax.text(0.04, 0.95, "252 / 823 cells differ", transform=ax.transAxes, fontsize=8.3, fontweight="bold", color=DARK, va="top", bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D6DDE1", "linewidth": 0.6})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(xlim=limits, ylim=limits, xlabel="Release endpoint A (cycles)", ylabel="Generator endpoint B (cycles)")
    ticks = [100, 300, 1000, 3000]
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_yticks(ticks, [str(value) for value in ticks])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(loc="lower right", handletextpad=0.4, borderaxespad=0.3)
    style_axis(ax, "both")
    panel(ax, "a", x=-0.09)

    ax = fig.add_subplot(gs[0, 3:])
    changed = cells[cells.labels_differ & cells.dataset_group.isin(["ISU_ILCC", "Tongji"])]
    groups = [changed.loc[changed.dataset_group == name, "label_delta_cycles"].abs().to_numpy() for name in ["ISU_ILCC", "Tongji"]]
    boxes = ax.boxplot(groups, positions=[0, 1], vert=False, widths=0.4, patch_artist=True, showfliers=False, medianprops={"color": DARK, "lw": 1.1}, whiskerprops={"color": MID, "lw": 0.7}, capprops={"color": MID, "lw": 0.7})
    for patch, face, edge in zip(boxes["boxes"], ["#F3DDCA", "#E6E1F0"], [ORANGE, PURPLE]):
        patch.set_facecolor(face)
        patch.set_edgecolor(edge)
    rng = np.random.default_rng(42)
    for position, (values, color, marker) in enumerate(zip(groups, ["#D99B65", "#978DB8"], ["o", "^"])):
        ax.scatter(values, position + rng.normal(0, 0.045, len(values)), s=6.0, color=color, marker=marker, linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 3600)
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_yticks([0, 1], ["ISU-ILCC", "Tongji"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute displacement |B − A| (cycles)")
    label_box = {"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none"}
    ax.text(0.02, 0.94, "Median |Δ|  9", transform=ax.transAxes, color=ORANGE, fontsize=6.9, fontweight="bold", va="top", bbox=label_box)
    ax.text(0.98, 0.08, "Median |Δ|  1", transform=ax.transAxes, color=PURPLE, fontsize=6.9, fontweight="bold", va="bottom", ha="right", bbox=label_box)
    style_axis(ax, "x")
    panel(ax, "b", x=-0.16)

    ax = fig.add_subplot(gs[1, 3:])
    values = np.array([0.8819, 2.992, 0.0])
    names = ["All", "ISU-ILCC", "Tongji"]
    colors = [NAVY, ORANGE, PURPLE]
    y = np.arange(3)
    for yi, value, color in zip(y, values, colors):
        ax.plot([0, value], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round")
        ax.scatter(value, yi, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        ax.text(max(value + 0.1, 0.12), yi, f"{value:.2f}%", va="center", fontsize=7.4, color=color, fontweight="bold")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set(xlim=(0, 3.55), xlabel="Comparable pairs reversed (%)")
    style_axis(ax, "x")
    panel(ax, "c", x=-0.16)

    ax = fig.add_subplot(gs[2, 3:])
    absolute = cells.loc[cells.labels_differ, "absolute_delta_cycles"].sort_values(ascending=False).to_numpy()
    x = np.arange(1, len(absolute) + 1) / len(absolute) * 100
    y = np.cumsum(absolute) / absolute.sum() * 100
    ax.fill_between(x, y, color="#F3D9D6")
    ax.plot(x, y, color=RED, linewidth=1.55)
    ax.plot([0, 100], [0, 100], linestyle=(0, (3, 3)), color=GREY, linewidth=0.7)
    for count in [1, 26]:
        ax.scatter(100 * count / len(absolute), y[count - 1], s=22, color=RED, edgecolor="white", linewidth=0.4, zorder=3)
    ax.annotate("1 cell\n23%", (100 / len(absolute), y[0]), xytext=(14, -3), textcoords="offset points", fontsize=7.1, color=RED, arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.6})
    ax.annotate("top 10%\n68%", (100 * 26 / len(absolute), y[25]), xytext=(8, 9), textcoords="offset points", fontsize=7.1, color=RED, arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.6})
    ax.set(xlabel="Changed cells, ranked by |Δ| (%)", ylabel="Cumulative |Δ| (%)", xlim=(0, 100), ylim=(0, 103))
    style_axis(ax, "both")
    panel(ax, "d", x=-0.16)
    save_figure(fig, 2, "Endpoint drift is source structured, rank changing within ISU-ILCC, and concentrated in a few cells.")


def figure3():
    status = pd.read_csv(DATA / "Fig3_status.csv")
    bootstrap = pd.read_csv(DATA / "Fig3_bootstrap.csv")
    shifts = pd.read_csv(DATA / "Fig3_shifts.csv")
    order = ["ridge", "xgboost", "cpmlp", "cptransformer"]
    labels = ["Ridge", "XGBoost", "CPMLP", "CPTransformer"]
    colors = [NAVY, BLUE, TEAL, PURPLE]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.45), constrained_layout=True, gridspec_kw={"width_ratios": [1.05, 1.7, 1.2], "wspace": 0.28})

    ax = axes[1]
    subset = bootstrap[bootstrap.metric == "unchanged_shift_share"].set_index("model").loc[order]
    y = np.arange(4)
    med = subset.bootstrap_median.to_numpy()
    low = subset.bootstrap_95_low.to_numpy()
    high = subset.bootstrap_95_high.to_numpy()
    ax.axvspan(0.5, 0.84, color="#EAF3F2", zorder=0)
    for index, color in enumerate(colors):
        ax.plot([low[index], high[index]], [y[index], y[index]], color=color, linewidth=1.45)
        ax.plot([low[index], low[index]], [y[index] - 0.06, y[index] + 0.06], color=color, linewidth=0.9)
        ax.plot([high[index], high[index]], [y[index] - 0.06, y[index] + 0.06], color=color, linewidth=0.9)
        ax.scatter(med[index], y[index], s=34, color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(high[index] + 0.012, y[index], f"{med[index] * 100:.1f}%", color=color, va="center", fontsize=7.5, fontweight="bold")
    ax.axvline(0.5, color=MID, linestyle=(0, (3, 3)), linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlim=(0.38, 0.86), xlabel="Share of absolute prediction movement\non unchanged-label test cells")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.text(0.66, 1.01, "more movement on unchanged labels →", transform=ax.transAxes, color=TEAL, fontsize=6.9, ha="center", va="bottom")
    style_axis(ax, "x")
    panel(ax, "b", x=-0.1)

    ax = axes[0]
    for index, (model, color) in enumerate(zip(order, colors)):
        values = status[(status.model == model) & (status.label_status == "changed")].evaluation_contribution_to_full_mean.to_numpy()
        ax.scatter(np.full(len(values), index) + np.linspace(-0.08, 0.08, len(values)), values, s=18, color=color, edgecolor="white", linewidth=0.35)
        ax.plot([index - 0.14, index + 0.14], [np.median(values)] * 2, color=DARK, linewidth=1.0)
    ax.axhline(0, color=MID, linewidth=0.65)
    ax.text(0.03, 0.05, "114 unchanged: exactly 0", transform=ax.transAxes, fontsize=6.7, color=MID, va="bottom")
    ax.set_xticks(range(4), ["R", "XGB", "MLP", "Tr"])
    ax.set(ylabel="Evaluation contribution", ylim=(-0.0006, 0.0152))
    style_axis(ax, "y")
    panel(ax, "a", x=-0.12)

    ax = axes[2]
    cpmlp = status[status.model == "cpmlp"].pivot(index="seed", columns="label_status", values="fitting_contribution_to_full_mean").loc[[42, 2021, 2024]]
    y = np.arange(3)
    for index, seed in enumerate([42, 2021, 2024]):
        unchanged = cpmlp.loc[seed, "unchanged"]
        changed = cpmlp.loc[seed, "changed"]
        ax.plot([0, unchanged], [index - 0.10, index - 0.10], color=TEAL, linewidth=1.6)
        ax.scatter(unchanged, index - 0.10, color=TEAL, s=24, marker="o")
        ax.plot([0, changed], [index + 0.10, index + 0.10], color=ORANGE, linewidth=1.6)
        ax.scatter(changed, index + 0.10, color=ORANGE, s=24, marker="s")
    ax.axvline(0, color=MID, linewidth=0.65)
    ax.set_yticks(y, ["42", "2021", "2024"])
    ax.invert_yaxis()
    ax.set_xlabel("CPMLP fitting contribution")
    ax.legend(handles=[Line2D([0], [0], color=TEAL, marker="o", label="Unchanged"), Line2D([0], [0], color=ORANGE, marker="s", label="Changed")], loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, handlelength=1.0, columnspacing=0.8)
    style_axis(ax, "x")
    panel(ax, "c", x=-0.12)
    save_figure(fig, 3, "Evaluation changes are local to relabelled cells, whereas fitting changes propagate to cells whose labels are unchanged.")


def figure4():
    data = pd.read_csv(DATA / "Fig4_metric_factor_summary.csv")
    order = ["ridge", "xgboost", "cpmlp", "cptransformer"]
    labels = ["Ridge", "XGBoost", "CPMLP", "CPTransformer"]
    metrics = ["MAPE", "MAE_cycles", "RMSE_cycles", "sMAPE", "absolute_log_error"]
    metric_labels = ["MAPE\n×100", "MAE\ncycles", "RMSE\ncycles", "sMAPE\n×100", "|log error|\n×100"]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("signed_v12", ["#476F91", "#F7F8F8", "#C4685E"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.55), sharey=True, gridspec_kw={"wspace": 0.08})
    fig.subplots_adjust(left=0.12, right=0.99, top=0.87, bottom=0.25)
    for ax, factor, letter, title in zip(axes, ["evaluation", "fitting"], ["a", "b"], ["Evaluation-label effect", "Fitting-label effect"]):
        raw = np.zeros((4, 5))
        for row, model in enumerate(order):
            for column, metric in enumerate(metrics):
                raw[row, column] = data.loc[(data.model == model) & (data.metric == metric), f"{factor}_median"].iloc[0]
        scaled = raw.copy()
        for column in range(5):
            scaled[:, column] /= max(abs(raw[:, column]).max(), 1e-12)
        ax.imshow(scaled, cmap=cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        for row in range(4):
            for column in range(5):
                value = raw[row, column] * 100 if metrics[column] in ["MAPE", "sMAPE", "absolute_log_error"] else raw[row, column]
                strong = abs(scaled[row, column]) > 0.63
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color="white" if strong else DARK, fontsize=7.5, fontweight="bold" if strong else "normal")
        ax.set_xticks(range(5), metric_labels)
        ax.set_yticks(range(4), labels)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.set_title(title, fontsize=8.5, fontweight="bold", color=DARK, pad=9)
        panel(ax, letter, x=-0.08 if letter == "a" else -0.04)
    fig.text(0.5, 0.055, "Blue: lower error under Contract B    ·    Red: higher error under Contract B    ·    colour normalized within each metric", ha="center", color=MID, fontsize=7.1)
    save_figure(fig, 4, "Endpoint propagation appears across several error measures, but effect direction is not invariant across models and routes.")


def figure5():
    transitions = pd.read_csv(DATA / "Fig5_interval_transitions.csv")
    margins = pd.read_csv(DATA / "Fig5_margin_components.csv")
    leverage = pd.read_csv(DATA / "Fig5_g34c2_leverage.csv")
    seeds = [42, 2021, 2024]
    fig = plt.figure(figsize=(6.8, 4.9))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.1, hspace=0.62, wspace=0.78)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.28])

    ax = fig.add_subplot(gs[0, :2])
    counts = transitions[transitions.dataset_group == "SNL"].pivot(index="seed", columns="transition", values="n_cells").fillna(0).loc[seeds]
    for row, seed in enumerate(seeds):
        both = int(counts.loc[seed].get("both", 0))
        b_only = int(counts.loc[seed].get("B_only", 0))
        neither = int(counts.loc[seed].get("neither", 0))
        colors = [TEAL] * both + [ORANGE] * b_only + [GREY] * neither
        markers = ["o"] * both + ["D"] * b_only + ["o"] * neither
        for column, (color, marker) in enumerate(zip(colors, markers), start=1):
            ax.scatter(column, row, s=55 if marker == "D" else 42, color=color, marker=marker, edgecolor="white", linewidth=0.55)
        ax.text(10.7, row, f"{both} / {b_only} / {neither}", fontsize=7.0, color=MID, va="center")
    ax.set_yticks(range(3), [str(seed) for seed in seeds])
    ax.set_xticks([1, 5, 10], ["1", "5", "10"])
    ax.invert_yaxis()
    ax.set(xlim=(0.3, 14.7), xlabel="SNL test cells", ylabel="Seed")
    for x0, marker, color, label in [(0.04, "o", TEAL, "both"), (0.36, "D", ORANGE, "B only"), (0.72, "o", GREY, "neither")]:
        ax.scatter(x0, 1.075, transform=ax.transAxes, clip_on=False, s=18, marker=marker, color=color, edgecolor="white", linewidth=0.35)
        ax.text(x0 + 0.04, 1.06, label, transform=ax.transAxes, color=color, fontsize=6.9, fontweight="bold")
    style_axis(ax, "x")
    panel(ax, "a", x=-0.14, y=1.06)

    ax = fig.add_subplot(gs[0, 2:])
    source = margins[margins.dataset_group == "SNL"].set_index("seed").loc[seeds]
    y = np.arange(3)
    for index, seed in enumerate(seeds):
        centre = source.loc[seed, "mean_centre_component"]
        radius = source.loc[seed, "mean_radius_component"]
        net = centre + radius
        ax.plot([0, centre], [index - 0.12, index - 0.12], color=RED, linewidth=1.8)
        ax.scatter(centre, index - 0.12, color=RED, s=28, marker="s")
        ax.plot([0, radius], [index + 0.12, index + 0.12], color=PURPLE, linewidth=1.8)
        ax.scatter(radius, index + 0.12, color=PURPLE, s=28, marker="o")
        ax.scatter(net, index, color=DARK, s=22, marker="D", zorder=3)
    ax.axvline(0, color=MID, linewidth=0.7)
    ax.set_yticks(y, [str(seed) for seed in seeds])
    ax.invert_yaxis()
    ax.set(xlabel="Mean change in interval margin (cycles)", ylabel="Seed")
    ax.set_ylim(2.55, -0.45)
    ax.text(0, 2.4, "target = 0", fontsize=6.7, color=MID, ha="center", va="top")
    for x0, marker, color, label in [(0.04, "s", RED, "Centre"), (0.34, "o", PURPLE, "Radius"), (0.64, "D", DARK, "Net margin")]:
        ax.scatter(x0, 1.075, transform=ax.transAxes, clip_on=False, s=18, marker=marker, color=color, edgecolor="white", linewidth=0.35)
        ax.text(x0 + 0.035, 1.06, label, transform=ax.transAxes, color=color, fontsize=6.9, fontweight="bold")
    style_axis(ax, "x")
    panel(ax, "b", x=-0.1, y=1.06)

    ax = fig.add_subplot(gs[1, :])
    data = leverage[leverage.dataset_group == "SNL"].copy()
    data["label"] = data.contract.map({"release": "A", "current_generator": "B"}) + " / " + data.seed.astype(str)
    index_order = [("release", 42), ("release", 2021), ("release", 2024), ("current_generator", 42), ("current_generator", 2021), ("current_generator", 2024)]
    data = data.set_index(["contract", "seed"]).loc[index_order].reset_index()
    y = np.arange(len(data))
    ax.axhspan(-0.5, 2.5, color="#EEF4F7", zorder=0)
    ax.axhspan(2.5, 5.5, color="#F5F1F8", zorder=0)
    for index, row in data.iterrows():
        ax.plot([0, row.median_absolute_prediction_shift], [index - 0.10, index - 0.10], color=TEAL, linewidth=1.75)
        ax.scatter(row.median_absolute_prediction_shift, index - 0.10, color=TEAL, s=27, marker="o", zorder=3)
        ax.plot([0, row.quantile_shift], [index + 0.10, index + 0.10], color=ORANGE, linewidth=1.75)
        ax.scatter(row.quantile_shift, index + 0.10, color=ORANGE, s=27, marker="s", zorder=3)
    ax.axvline(0, color=MID, linewidth=0.7)
    ax.set_yticks(y, data.label)
    ax.invert_yaxis()
    ax.set(xlabel="Change after G34C2 removal and refitting (cycles)", xlim=(-210, 430), ylim=(5.55, -0.55))
    ax.legend(handles=[Line2D([0], [0], color=TEAL, marker="o", label="Median |prediction shift|"), Line2D([0], [0], color=ORANGE, marker="s", label="Interval-radius shift")], loc="upper center", bbox_to_anchor=(0.53, 1.14), ncol=2, handlelength=1.4, columnspacing=1.2)
    ax.text(0.985, 0.045, "G34C2 endpoint:  A 3109  →  B 317 cycles", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2, color=RED, fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"})
    style_axis(ax, "x")
    panel(ax, "c", x=-0.045, y=1.06)
    save_figure(fig, 5, "The SNL coverage difference is one boundary crossing per seed, caused by radius expansion rather than relabelling and sensitive to a high-leverage training record.")


def write_manifest():
    records = []
    for path in sorted(OUT.glob("Fig[1-5].*")):
        if path.suffix not in {".svg", ".pdf", ".eps", ".png"}:
            continue
        records.append({"file": path.name, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    catalogue = [
        {"figure": 1, "surface": "paper_main", "claim": "Endpoint execution affects a fixed benchmark through four parallel pathways.", "source_data": [], "self_review_focus": "parallel rather than sequential route logic"},
        {"figure": 2, "surface": "paper_main", "claim": "Endpoint displacement is source structured and concentrated.", "source_data": ["figure_data/Fig2_target_geometry_cells.csv"], "self_review_focus": "hero geometry panel and legible extreme cell"},
        {"figure": 3, "surface": "paper_main", "claim": "Fitting changes propagate beyond relabelled cells.", "source_data": ["figure_data/Fig3_status.csv", "figure_data/Fig3_bootstrap.csv", "figure_data/Fig3_shifts.csv"], "self_review_focus": "evaluation and fitting routes remain distinct"},
        {"figure": 4, "surface": "paper_main", "claim": "Effect direction depends on model, metric, and route.", "source_data": ["figure_data/Fig4_metric_factor_summary.csv"], "self_review_focus": "exact values remain readable despite normalized colour"},
        {"figure": 5, "surface": "paper_main", "claim": "One SNL boundary crossing is driven by radius expansion and is leverage sensitive.", "source_data": ["figure_data/Fig5_interval_transitions.csv", "figure_data/Fig5_margin_components.csv", "figure_data/Fig5_g34c2_leverage.csv"], "self_review_focus": "one-cell transition is visually immediate"},
    ]
    (OUT / "figure_catalog_v12.json").write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    payload = {
        "status": "PASS" if len(records) == 20 else "FAIL",
        "backend": "Python/matplotlib",
        "figure_files": records,
        "source_tables": sorted(path.name for path in DATA.glob("*.csv")),
        "style": {"font": "Arial-compatible sans serif", "minimum_nominal_font_pt": 7.0, "primary_format": "SVG", "png_dpi": 600, "width_in": 6.8},
    }
    (OUT / "figure_manifest_v12.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    global DATA, OUT
    parser = argparse.ArgumentParser(description="Build the five paper figures from derived source tables.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    DATA = args.data
    OUT = args.output
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    write_manifest()


if __name__ == "__main__":
    main()
