import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from verify_public_repo import verify

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "02_experiments/analysis"))
import analyze_reviewer_experiments_v13 as analysis

MAIN = Path("04_analysis/reviewer_v13_20260904")
META = Path("04_analysis/reviewer_v14_20260905/metadata_ablation")
INPUT = Path("03_data_interaction/model_input/gate1_v2_20260821_0656")
PAIRED = Path("03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv")


def run(script, *arguments):
    print(f"Running {Path(script).name}", flush=True)
    subprocess.run([sys.executable, str(REPO / script), *map(str, arguments)], check=True,
                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"})


def metadata(data, output):
    output.mkdir()
    classical = pd.read_csv(data / META / "classical_blind_predictions.csv")
    original = pd.read_csv(data / MAIN / "core/blind_predictions.csv")
    deep = original[original.model.isin(("cpmlp", "cptransformer"))]
    combined = pd.concat([classical, deep], ignore_index=True)
    combined.to_csv(output / "combined_blind_predictions.csv", index=False)
    labels = analysis.load_labels(data / INPUT / "labels")
    paired = analysis.paired_metadata(data / PAIRED)
    _, points, factors = analysis.point_outputs(combined, labels)
    ranks, comparisons, best, pairwise = analysis.ranking_outputs(points)
    wide, cross, within, variation = analysis.refit_variability(combined, paired)
    print("Metadata ablation: 5,000 paired seed/source-cell bootstrap replicates", flush=True)
    replicates, summary = analysis.prediction_shift_bootstrap(wide, 5000)
    outputs = {"point_metrics_by_seed.csv": points, "factor_contrasts_by_seed.csv": factors,
               "model_ranks_by_cell.csv": ranks, "ranking_comparisons.csv": comparisons,
               "best_model_frequency.csv": best, "pairwise_model_wins.csv": pairwise,
               "prediction_shift_cells.csv": wide, "cross_contract_prediction_shift_by_seed.csv": cross,
               "within_contract_refit_pairwise.csv": within, "refit_variability_summary.csv": variation,
               "prediction_shift_bootstrap_replicates.csv": replicates,
               "prediction_shift_bootstrap_summary.csv": summary}
    reference = pd.read_csv(data / MAIN / "analysis/prediction_shift_bootstrap_summary.csv")
    columns = ["model", "label_status", "bootstrap_median", "bootstrap_95_low", "bootstrap_95_high"]
    selected = lambda frame: frame[frame.model.eq("xgboost") & frame.quantity.eq("cross_to_within_ratio")][columns]
    outputs["xgboost_ratio_comparison.csv"] = selected(summary).merge(
        selected(reference), on=["model", "label_status"], suffixes=("_early_only", "_metadata_aware"))
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)


def main():
    parser = argparse.ArgumentParser(description="Recompute v14 results and figures from the derived-data release.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data, output = args.data_root.resolve(), args.output.resolve()
    verify(REPO)
    verify(data)
    if output == data or output.is_relative_to(data) or data.is_relative_to(output):
        raise SystemExit("Output and source data must be separate directories.")
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Refusing to overwrite a non-empty output directory.")
    output.mkdir(parents=True, exist_ok=True)
    core = data / MAIN / "core"
    baseline = data / "03_data_interaction/baseline/gate1_v2_20260821_0656"
    run("02_experiments/analysis/analyze_reviewer_experiments_v13.py",
        "--predictions", core / "blind_predictions.csv", "--training-log", core / "training_runs.jsonl",
        "--frozen-configuration", core / "frozen_configuration.json", "--labels-root", data / INPUT / "labels",
        "--paired-labels", data / PAIRED, "--original-release-predictions", baseline / "release/revealed_predictions.csv",
        "--original-current-predictions", baseline / "current_generator/revealed_predictions.csv",
        "--output", output / "main")
    metadata(data, output / "metadata")
    run("scripts/validate_replay.py", "--data-root", data, "--results", output,
        "--output", output / "verification.json")
    run("02_experiments/analysis/audit_rank_gaps_v14.py", "--data-root", data,
        "--analysis-root", output / "main", "--output", output / "rank_gap")
    run("02_experiments/analysis/audit_tongji_exception_v14.py", "--data-root", data,
        "--output", output / "tongji_trace")
    run("02_experiments/data_integrity/audit_isu_contract_coordinates.py", "--data-root", data,
        "--output", output / "isu_coordinates")
    run("06_figures/reviewer_v14_sync/build_jeet_figures_v14.py", "--data-root", data,
        "--analysis-root", output / "main", "--output", output / "figures")
    receipt = {"status": "PASS", "scope": "stored-prediction replay and derived-trace validation",
               "model_training_performed": False, "bootstrap_replicates": 5000,
               "code_manifest_sha256": analysis.sha256(REPO / "MANIFEST.json"),
               "data_manifest_sha256": analysis.sha256(data / "MANIFEST.json")}
    (output / "reproduction.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
