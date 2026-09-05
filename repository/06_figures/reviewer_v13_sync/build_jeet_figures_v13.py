from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd


plt.style.use(Path(__file__).with_name("academic_v13.mplstyle"))
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "legend.fontsize": 8,
})

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LEGACY = ROOT / "04_analysis/depth_v10_20260901/core"
V13 = ROOT / "04_analysis/reviewer_v13_20260904/analysis"
COORD = ROOT / "03_data_interaction/data_integrity/isu_contract_coordinate_audit/contract_coordinate_summary.json"
DATA = OUT / "figure_data"
VERSION = "v13"


def configure(data_root, output, analysis_root=None):
    global ROOT, OUT, LEGACY, V13, COORD, DATA
    ROOT = data_root.resolve()
    OUT = output.resolve()
    if OUT.is_relative_to(ROOT) or (OUT.exists() and any(OUT.iterdir())):
        raise ValueError("--output must be a new or empty directory outside --data-root")
    LEGACY = ROOT / "04_analysis/depth_v10_20260901/core"
    V13 = analysis_root.resolve() if analysis_root else ROOT / "04_analysis/reviewer_v13_20260904/analysis"
    COORD = ROOT / "03_data_interaction/data_integrity/isu_contract_coordinate_audit/contract_coordinate_summary.json"
    DATA = OUT / "figure_data"
    DATA.mkdir(parents=True)

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

MODEL_ORDER = ["ridge", "xgboost", "cpmlp", "cptransformer"]
MODEL_LABEL = {
    "ridge": "Ridge",
    "xgboost": "XGBoost",
    "cpmlp": "CPMLP",
    "cptransformer": "CPTransformer",
}
MODEL_COLOR = {
    "ridge": NAVY,
    "xgboost": ORANGE,
    "cpmlp": TEAL,
    "cptransformer": PURPLE,
}
METRIC_ORDER = ["mape", "smape", "mae", "rmse", "log_mae"]
METRIC_LABEL = {"mape": "MAPE", "smape": "sMAPE", "mae": "MAE", "rmse": "RMSE", "log_mae": "log-MAE"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style_axis(ax, grid=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.8, width=0.7, color=MID)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.55)
        ax.set_axisbelow(True)


def panel(ax, label, x=-0.08, y=1.03):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.2, fontweight="bold", color=DARK, ha="left", va="bottom")


def save_figure(fig, number, conclusion):
    base = OUT / f"Fig{number}"
    fig.savefig(base.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(base.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(base.with_suffix(".eps"), metadata={"Creator": "Matplotlib"})
    fig.savefig(base.with_suffix(".png"), dpi=600, metadata={"Software": "Matplotlib"})
    plt.close(fig)
    (OUT / f"Fig{number}_conclusion.txt").write_text(conclusion + "\n", encoding="utf-8")


def card(ax, x, y, w, h, title, body, edge, dashed=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor=edge,
        linewidth=1.0,
        linestyle=(0, (3, 2)) if dashed else "solid",
    )
    ax.add_patch(patch)
    ax.text(x + 0.018, y + h - 0.055, title, transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=DARK, va="top")
    ax.text(x + 0.018, y + h - 0.145, body, transform=ax.transAxes, fontsize=8.0, color=MID, va="top", linespacing=1.3)


def figure1():
    coord = json.loads(COORD.read_text(encoding="utf-8"))
    rows = [
        {
            "quantity_id": "source_author_endpoint",
            "display_label": "Author endpoint",
            "observation_schedule": "periodic reference-performance tests",
            "coordinate": "cumulative calendar time",
            "unit": "weeks",
            "model_target": False,
            "provenance_status": "lineage diagnostic only; no conversion to model labels",
        },
        {
            "quantity_id": "contract_a",
            "display_label": "Contract A",
            "observation_schedule": "retained processed ordinary-cycle sequence",
            "coordinate": "one-based retained position",
            "unit": "processed-position units",
            "model_target": True,
            "provenance_status": f"exact post hoc effective reconstruction {coord['effective_a_reconstruction_exact']}; historical execution unrecovered",
        },
        {
            "quantity_id": "contract_b",
            "display_label": "Contract B",
            "observation_schedule": "retained processed ordinary-cycle sequence",
            "coordinate": "one-based retained position",
            "unit": "processed-position units",
            "model_target": True,
            "provenance_status": "pinned-generator output",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA / "Fig1_coordinate_contract.csv", index=False)

    fig = plt.figure(figsize=(6.8, 5.2))
    ax = fig.add_axes([0.02, 0.48, 0.96, 0.49])
    ax.axis("off")
    panel(ax, "a", x=0.0, y=0.98)
    card(ax, 0.03, 0.48, 0.27, 0.37, "Author endpoint", "Periodic RPT schedule\nCumulative calendar weeks\nLineage diagnostic only", MID, dashed=True)
    card(ax, 0.365, 0.48, 0.27, 0.37, "Contract A", "Archived label artifact\nExact post hoc reconstruction\nHistorical execution unrecovered", BLUE)
    card(ax, 0.70, 0.48, 0.27, 0.37, "Contract B", "Pinned-generator output\nVersion-controlled rule\nSame processed sequence", ORANGE)
    ax.add_patch(FancyArrowPatch((0.50, 0.46), (0.50, 0.31), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, color=BLUE, linewidth=0.9))
    ax.add_patch(FancyArrowPatch((0.835, 0.46), (0.835, 0.31), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=9, color=ORANGE, linewidth=0.9))
    ax.add_patch(FancyBboxPatch((0.35, 0.10), 0.52, 0.20, boxstyle="round,pad=0.012,rounding_size=0.018", transform=ax.transAxes, facecolor=LIGHT, edgecolor=NAVY, linewidth=1.0))
    ax.text(0.61, 0.22, "Shared benchmark coordinate", transform=ax.transAxes, ha="center", va="center", fontsize=8.4, fontweight="bold", color=DARK)
    ax.text(0.61, 0.135, "integer, one-based position on the retained processed ordinary-cycle sequence", transform=ax.transAxes, ha="center", va="center", fontsize=8.0, color=MID)
    ax.text(0.165, 0.20, "not a model target", transform=ax.transAxes, ha="center", va="center", fontsize=8.0, color=MID)

    ax2 = fig.add_axes([0.03, 0.06, 0.94, 0.36])
    ax2.axis("off")
    panel(ax2, "b", x=-0.01, y=1.02)
    ax2.text(0.5, 0.94, "Fixed cohort, inputs, split, and explicit model configuration", transform=ax2.transAxes, ha="center", fontsize=8.4, fontweight="bold", color=DARK)
    labels = [
        ("Target geometry", "823 paired cells\n252 changed labels"),
        ("2 × 2 contrasts", "fit and evaluation\nroutes separated"),
        ("Refit baseline", "23 seed indices\nwithin vs cross"),
        ("Ranking", "winner, top two,\nfull order"),
        ("Conformal", "coverage, efficiency,\nquantile provenance"),
    ]
    xs = np.linspace(0.0, 0.80, 5)
    for i, (x, (head, body)) in enumerate(zip(xs, labels)):
        card(ax2, x, 0.23, 0.18, 0.50, head, body, [BLUE, TEAL, ORANGE, PURPLE, RED][i])
        if i < 4:
            ax2.add_patch(FancyArrowPatch((x + 0.182, 0.48), (x + 0.198, 0.48), transform=ax2.transAxes, arrowstyle="-|>", mutation_scale=8, color=MID, linewidth=0.8))
    save_figure(fig, 1, "The author week endpoint is a separate lineage diagnostic; Contracts A and B are aligned processed-position labels entering the benchmark through parallel analytical routes.")


def figure2():
    d = pd.read_csv(LEGACY / "target_geometry_cells.csv")
    d.to_csv(DATA / "Fig2_target_geometry_cells.csv", index=False)
    changed = d[d["labels_differ"]].copy()
    summary = pd.read_csv(LEGACY / "target_geometry_summary.csv")
    source_rank = summary[summary["group_level"].eq("source")].set_index("group")

    fig = plt.figure(figsize=(6.8, 5.4))
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.98, bottom=0.10, top=0.96, hspace=0.38, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(d["release_eol"], d["current_generator_eol"], s=8, color=GREY, alpha=0.38, linewidths=0)
    for group, color in [("Tongji", TEAL), ("ISU_ILCC", ORANGE)]:
        g = changed[changed["dataset_group"].eq(group)]
        ax.scatter(g["release_eol"], g["current_generator_eol"], s=12, color=color, alpha=0.78, linewidths=0, label=group.replace("_", "-"))
    low = min(d["release_eol"].min(), d["current_generator_eol"].min())
    high = max(d["release_eol"].max(), d["current_generator_eol"].max())
    ax.plot([low, high], [low, high], color=DARK, linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log = FuncFormatter(lambda value, _: f"{value:,.0f}")
    ax.xaxis.set_major_formatter(plain_log)
    ax.yaxis.set_major_formatter(plain_log)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Contract A retained processed position")
    ax.set_ylabel("Contract B retained processed position")
    ax.legend(loc="lower right")
    style_axis(ax)
    panel(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    groups = ["Tongji", "ISU_ILCC"]
    vals = [changed.loc[changed.dataset_group.eq(g), "absolute_delta_cycles"].to_numpy() for g in groups]
    parts = ax.violinplot(vals, positions=[0, 1], showmedians=True, showextrema=False, widths=0.72)
    for body, color in zip(parts["bodies"], [TEAL, ORANGE]):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor("none")
    parts["cmedians"].set_color(DARK)
    parts["cmedians"].set_linewidth(1.0)
    ax.set_xticks([0, 1], ["Tongji", "ISU-ILCC"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plain_log)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Absolute A–B displacement\n(processed-position units)")
    ax.text(0, np.nanmax(vals[0]) * 1.4, "median 1", ha="center", fontsize=8, color=TEAL)
    ax.text(1, np.nanpercentile(vals[1], 90) * 1.25, "median 9; P90 112", ha="center", fontsize=8, color=ORANGE)
    style_axis(ax, "y")
    panel(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    rates = [100 * source_rank.loc[g, "discordant_pair_fraction"] for g in groups]
    bars = ax.bar(groups, rates, color=[TEAL, ORANGE], width=0.58)
    ax.set_xticks([0, 1], ["Tongji", "ISU-ILCC"])
    ax.set_ylabel("Pairwise rank reversals (%)")
    ax.set_ylim(0, max(rates) * 1.25 + 0.05)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.04, f"{rate:.2f}%", ha="center", va="bottom", fontsize=8)
    style_axis(ax, "y")
    panel(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    top = changed.nlargest(8, "absolute_delta_cycles").sort_values("absolute_delta_cycles")
    colors = [ORANGE if g == "ISU_ILCC" else TEAL for g in top["dataset_group"]]
    ax.barh(np.arange(len(top)), top["absolute_delta_cycles"], color=colors)
    short = [name.replace(".pkl", "").split("_")[-1] for name in top["file_name"]]
    ax.set_yticks(np.arange(len(top)), short)
    ax.set_xlabel("Absolute displacement\n(processed-position units)")
    ax.tick_params(axis="y", labelsize=8)
    style_axis(ax, "x")
    panel(ax, "d")
    save_figure(fig, 2, "Contract A–B displacement on the shared processed-position coordinate is source structured and concentrated in a small number of cells.")


def figure3():
    point = pd.read_csv(V13 / "refit_variability_summary.csv")
    boot = pd.read_csv(V13 / "prediction_shift_bootstrap_summary.csv")
    boot = boot[boot["quantity"].eq("cross_to_within_ratio")].copy()
    point.to_csv(DATA / "Fig3_refit_point.csv", index=False)
    boot.to_csv(DATA / "Fig3_refit_bootstrap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), gridspec_kw={"width_ratios": [0.92, 1.28]})
    ax = axes[0]
    all_rows = point[point["label_status"].eq("all")].set_index("model").loc[MODEL_ORDER]
    y = np.arange(len(MODEL_ORDER))[::-1]
    for yi, model in zip(y, MODEL_ORDER):
        row = all_rows.loc[model]
        cross = row["cross_contract_seed_median_mean_absolute_shift"]
        within = row["within_pooled_pair_median_mean_absolute_shift"]
        ax.plot([within, cross], [yi, yi], color=MODEL_COLOR[model], alpha=0.65, linewidth=1.2)
        ax.scatter(within, yi, marker="o", facecolor="white", edgecolor=MODEL_COLOR[model], s=32, linewidth=1.1, zorder=3)
        ax.scatter(cross, yi, marker="s", color=MODEL_COLOR[model], s=28, zorder=3)
    ax.set_yticks(y, [MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_xlabel("Mean absolute prediction shift\n(processed-position units)")
    style_axis(ax, "x")
    ax.set_title("All test cells\n■ cross; ○ within; Ridge within = 0", fontsize=8.4, fontweight="bold")
    panel(ax, "a")

    ax = axes[1]
    status_order = ["all", "changed", "unchanged"]
    offsets = {"xgboost": 0.22, "cpmlp": 0.0, "cptransformer": -0.22}
    for model in ["xgboost", "cpmlp", "cptransformer"]:
        b = boot[boot["model"].eq(model)].set_index("label_status")
        for base, status in enumerate(status_order[::-1]):
            row = b.loc[status]
            yi = base + offsets[model]
            ax.plot([row["bootstrap_95_low"], row["bootstrap_95_high"]], [yi, yi], color=MODEL_COLOR[model], linewidth=1.2)
            ax.scatter(row["bootstrap_median"], yi, color=MODEL_COLOR[model], s=25, zorder=3)
    ax.axvline(1, color=DARK, linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(3), ["Unchanged (114)", "Changed (44)", "All (158)"])
    ax.set_xlabel("Cross-contract / within-contract shift ratio")
    ax.set_xlim(0.15, 2.02)
    style_axis(ax, "x")
    ax.legend(handles=[Line2D([0], [0], marker="o", color=MODEL_COLOR[m], label=MODEL_LABEL[m], markersize=4) for m in ["xgboost", "cpmlp", "cptransformer"]], loc="upper right")
    ax.set_title("Stochastic models; Ridge ratio undefined", fontsize=8.4, fontweight="bold")
    panel(ax, "b")
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.91, wspace=0.38)
    save_figure(fig, 3, "Contract-induced movement exceeds ordinary refit variability only for XGBoost on changed-label cells; both deep-model ratios remain below one.")


def figure4():
    best = pd.read_csv(V13 / "best_model_frequency.csv")
    ranks = pd.read_csv(V13 / "ranking_comparisons.csv")
    matched = ranks[ranks["direction"].eq("matched_contract")].copy()
    stability = (
        matched.groupby("metric", as_index=False)
        .agg(
            n_comparisons=("seed", "size"),
            winner_same_count=("best_model_changed", lambda x: int((~x).sum())),
            top2_same_count=("top2_set_same", "sum"),
            full_order_same_count=("full_order_same", "sum"),
        )
    )
    totals = pd.DataFrame([{
        "metric": "ALL",
        "n_comparisons": len(matched),
        "winner_same_count": int((~matched["best_model_changed"]).sum()),
        "top2_same_count": int(matched["top2_set_same"].sum()),
        "full_order_same_count": int(matched["full_order_same"].sum()),
    }])
    stability = pd.concat([stability, totals], ignore_index=True)
    for name in ["winner", "top2", "full_order"]:
        stability[f"{name}_same_fraction"] = stability[f"{name}_same_count"] / stability["n_comparisons"]
    best.to_csv(DATA / "Fig4_best_frequency.csv", index=False)
    stability.to_csv(DATA / "Fig4_rank_stability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.7), gridspec_kw={"width_ratios": [1.28, 1.0]})
    ax = axes[0]
    cols = [(cell, metric) for cell in ["AA", "BB"] for metric in METRIC_ORDER]
    matrix = np.zeros((len(MODEL_ORDER), len(cols)))
    for i, model in enumerate(MODEL_ORDER):
        for j, (cell, metric) in enumerate(cols):
            matrix[i, j] = best.loc[(best.cell.eq(cell)) & (best.metric.eq(metric)) & (best.model.eq(model)), "best_seed_count"].iloc[0]
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=23, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=8, color="white" if matrix[i, j] > 13 else DARK)
    ax.set_xticks(np.arange(len(cols)), [METRIC_LABEL[m] for _, m in cols], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.axvline(4.5, color="white", linewidth=2.0)
    ax.text(2, -0.78, "AA", ha="center", fontsize=8.4, fontweight="bold", color=DARK)
    ax.text(7, -0.78, "BB", ha="center", fontsize=8.4, fontweight="bold", color=DARK)
    ax.set_xlabel("Best-model frequency out of 23 seeds")
    panel(ax, "a")

    ax = axes[1]
    st = stability[stability["metric"].ne("ALL")].set_index("metric").loc[METRIC_ORDER]
    y = np.arange(len(METRIC_ORDER))
    specs = [
        ("winner_same_count", "Winner", NAVY, 0.24),
        ("top2_same_count", "Top-two set", TEAL, 0.00),
        ("full_order_same_count", "Full order", ORANGE, -0.24),
    ]
    for col, label, color, offset in specs:
        vals = st[col].to_numpy()
        ax.barh(y + offset, 100 * vals / 23, height=0.20, color=color, label=label)
        for yi, value in zip(y + offset, vals):
            ax.text(100 * value / 23 - 1.5, yi, f"{int(value)}/23", ha="right", va="center", fontsize=8, color="white" if value >= 18 else DARK)
    ax.set_yticks(y, [METRIC_LABEL[m] for m in METRIC_ORDER])
    ax.set_xlim(55, 102)
    ax.set_xlabel("Matched-contract agreement (%)")
    fig.text(0.64, 0.94, "■ Winner", color=NAVY, fontsize=8.4, ha="left")
    fig.text(0.76, 0.94, "■ Top-two", color=TEAL, fontsize=8.4, ha="left")
    fig.text(0.89, 0.94, "■ Full order", color=ORANGE, fontsize=8.4, ha="left")
    style_axis(ax, "x")
    panel(ax, "b")
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.82, wspace=0.35)
    save_figure(fig, 4, "The winning model is stable across all 115 matched comparisons, whereas top-two membership and lower-rank order remain metric sensitive.")


def figure5():
    raw = pd.read_csv(V13 / "conformal_metrics.csv")
    pooled = raw[(raw["population"].eq("ALL")) & (raw["calibration_ablation"].eq("none"))]
    efficiency = (
        pooled.groupby(["contract", "model", "nominal"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_calibration=("n_calibration", "first"),
            coverage_median=("coverage", "median"),
            mean_width_median=("mean_width", "median"),
            normalized_mean_width_common_train_iqr_median=("normalized_mean_width_common_train_iqr", "median"),
            mean_interval_score_median=("mean_interval_score", "median"),
            calibration_quantile_median=("calibration_quantile", "median"),
            order_statistic=("order_statistic", "first"),
        )
    )
    rows = []
    base = pooled.groupby(["contract", "model", "nominal"], as_index=False).agg(base_radius_median=("calibration_quantile", "median"), n_seeds=("seed", "nunique"))
    for ablation, source in [("drop_ISU_ILCC", "ISU-ILCC"), ("drop_Tongji", "Tongji")]:
        drop = raw[(raw["population"].eq("ALL")) & (raw["calibration_ablation"].eq(ablation))]
        drop = drop.groupby(["contract", "model", "nominal"], as_index=False).agg(ablated_radius_median=("calibration_quantile", "median"))
        merged = base.merge(drop, on=["contract", "model", "nominal"], validate="one_to_one")
        merged["calibration_ablation"] = ablation
        merged["excluded_calibration_source"] = source
        merged["radius_percent_change"] = (merged["ablated_radius_median"] / merged["base_radius_median"] - 1) * 100
        rows.append(merged)
    removal = pd.concat(rows, ignore_index=True)
    removal = removal[[
        "contract", "model", "nominal", "n_seeds", "calibration_ablation",
        "excluded_calibration_source", "base_radius_median", "ablated_radius_median",
        "radius_percent_change",
    ]]
    efficiency.to_csv(DATA / "Fig5_conformal_efficiency.csv", index=False)
    removal.to_csv(DATA / "Fig5_source_removal.csv", index=False)

    fig = plt.figure(figsize=(6.8, 5.7))
    gs = fig.add_gridspec(2, 2, left=0.14, right=0.98, bottom=0.11, top=0.96, height_ratios=[1.0, 1.12], hspace=0.46, wspace=0.32)
    for col, nominal in enumerate([0.90, 0.95]):
        ax = fig.add_subplot(gs[0, col])
        sub = efficiency[np.isclose(efficiency["nominal"], nominal)]
        for model in MODEL_ORDER:
            g = sub[sub["model"].eq(model)].set_index("contract")
            x = [g.loc[c, "normalized_mean_width_common_train_iqr_median"] for c in ["release", "current_generator"]]
            y = [g.loc[c, "coverage_median"] for c in ["release", "current_generator"]]
            ax.plot(x, y, color=MODEL_COLOR[model], linewidth=0.9, alpha=0.7)
            ax.scatter(x[0], y[0], s=30, marker="o", color=MODEL_COLOR[model], label=MODEL_LABEL[model] if col == 0 else None)
            ax.scatter(x[1], y[1], s=32, marker="s", color=MODEL_COLOR[model])
        ax.axhline(nominal, color=DARK, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Mean width / common train IQR")
        ax.set_ylabel("Median pooled coverage")
        ax.set_title(f"{int(nominal * 100)}% nominal; k = {147 if nominal == .90 else 155}", fontsize=8.4, fontweight="bold")
        style_axis(ax, "both")
        panel(ax, "a" if col == 0 else "b")
        if col == 0:
            ax.legend(loc="lower right")
    fig.axes[0].text(0.02, 0.03, "○ A   ■ B", transform=fig.axes[0].transAxes, fontsize=8, color=MID)

    ax = fig.add_subplot(gs[1, :])
    sub = removal[np.isclose(removal["nominal"], 0.90)].copy()
    combos = [(model, contract) for model in MODEL_ORDER for contract in ["release", "current_generator"]]
    y = np.arange(len(combos))
    for source, color, offset in [("ISU-ILCC", ORANGE, 0.18), ("Tongji", TEAL, -0.18)]:
        values = []
        for model, contract in combos:
            values.append(sub.loc[(sub.model.eq(model)) & (sub.contract.eq(contract)) & (sub.excluded_calibration_source.eq(source)), "radius_percent_change"].iloc[0])
        ax.barh(y + offset, values, height=0.30, color=color, label=f"Remove {source}")
    ax.axvline(0, color=DARK, linewidth=0.8)
    short_model = {"ridge": "Ridge", "xgboost": "XGB", "cpmlp": "CPMLP", "cptransformer": "CPT"}
    ax.set_yticks(y, [f"{short_model[m]} · {'A' if c == 'release' else 'B'}" for m, c in combos])
    ax.set_xlabel("Change in median 90% calibration radius (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    style_axis(ax, "x")
    panel(ax, "c", x=-0.04)
    save_figure(fig, 5, "Pooled conformal efficiency is model dependent, and calibration radii are sensitive to source composition without a retained-sample or source-conditional guarantee.")


def write_metadata():
    catalogue = {
        "version": VERSION,
        "figures": [
            {"number": 1, "message": "The author week endpoint and aligned Contracts A and B.", "source_data": ["Fig1_coordinate_contract.csv"]},
            {"number": 2, "message": "Source-structured label displacement on the shared processed-position coordinate.", "source_data": ["Fig2_target_geometry_cells.csv"]},
            {"number": 3, "message": "Contract movement relative to ordinary refit variability.", "source_data": ["Fig3_refit_point.csv", "Fig3_refit_bootstrap.csv"]},
            {"number": 4, "message": "Stable winner selection and lower-rank instability.", "source_data": ["Fig4_best_frequency.csv", "Fig4_rank_stability.csv"]},
            {"number": 5, "message": "Conformal efficiency and calibration-composition sensitivity.", "source_data": ["Fig5_conformal_efficiency.csv", "Fig5_source_removal.csv"]},
        ],
    }
    (OUT / f"figure_catalog_{VERSION}.json").write_text(json.dumps(catalogue, indent=2) + "\n", encoding="utf-8")
    source_files = [
        COORD,
        LEGACY / "target_geometry_cells.csv",
        LEGACY / "target_geometry_summary.csv",
        V13 / "refit_variability_summary.csv",
        V13 / "prediction_shift_bootstrap_summary.csv",
        V13 / "best_model_frequency.csv",
        V13 / "ranking_comparisons.csv",
        V13 / "conformal_metrics.csv",
    ]
    output_files = [OUT / f"Fig{i}.{ext}" for i in range(1, 6) for ext in ["svg", "pdf", "eps", "png"]]
    figure_data = sorted(DATA.glob("*.csv"))
    manifest = {
        "status": "PASS",
        "version": VERSION,
        "nominal_min_font_pt": 8.4 if VERSION == "v14" else 8.0,
        "source_hashes": {p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else "analysis/" + p.name: sha256(p) for p in source_files},
        "output_hashes": {p.name: sha256(p) for p in output_files},
        "figure_data_hashes": {p.name: sha256(p) for p in figure_data},
        "figures": catalogue["figures"],
    }
    (OUT / f"figure_manifest_{VERSION}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build the shared figure data and base layouts.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure(args.data_root, args.output, args.analysis_root)
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    write_metadata()
    print("PASS: generated five v13 figures in SVG, PDF, EPS, and 600-dpi PNG formats")


if __name__ == "__main__":
    main()
