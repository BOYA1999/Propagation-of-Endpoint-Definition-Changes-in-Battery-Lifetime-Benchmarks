import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import run_controlled_refits_v13 as controlled

import numpy as np
import pandas as pd
import psutil
import sklearn
import xgboost

import analyze_reviewer_experiments_v13 as analysis

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT
OUT = ROOT / "04_analysis/reviewer_v14_20260905/metadata_ablation"
FIT_RESULTS = OUT
INPUT = ROOT / "03_data_interaction/model_input/gate1_v2_20260821_0656"
CORE = ROOT / "04_analysis/reviewer_v13_20260904/core"
BASE = CORE.parent / "analysis"
SEEDS = controlled.DEFAULT_SEEDS


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def relative(path):
    path = path.resolve()
    for name, base in (("output", OUT), ("fit_results", FIT_RESULTS), ("code", CODE_ROOT), ("data", ROOT)):
        if path.is_relative_to(base.resolve()):
            return name + "/" + path.relative_to(base.resolve()).as_posix()
    raise ValueError(f"Input is outside declared roots: {path.name}")


def resource_receipt(start):
    memory = psutil.Process().memory_info()
    metric = {"resident_set_bytes": memory.rss}
    if hasattr(memory, "peak_wset"):
        metric["peak_working_set_bytes"] = memory.peak_wset
    return {"seconds": time.perf_counter() - start, **metric,
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}


def fit():
    start = time.perf_counter()
    if (OUT / "training_runs.jsonl").exists() or (OUT / "classical_blind_predictions.csv").exists():
        raise SystemExit("Existing training artifacts retained; use the saved run rather than overwrite")
    paths = [INPUT / "cycle100_features.csv"] + [INPUT / "labels" / c / "train_labels.csv" for c in controlled.CONTRACTS]
    features = pd.read_csv(paths[0])
    numeric = [c for c in features if c not in ("split", "file_name", "dataset_group", "chemistry_system")]
    assert len(numeric) == 126 and features.groupby("split").size().to_dict() == {"calibration": 162, "test": 158, "train": 503}
    labels = controlled.validate_labels(features, dict(zip(controlled.CONTRACTS, paths[1:])))
    config = {"created_utc": now(), "protocol_sha256": analysis.sha256(ROOT / "04_analysis/reviewer_v14_20260905/metadata_ablation/protocol.md"),
              "source_sha256": analysis.sha256(Path(__file__)),
              "controlled_source_sha256": analysis.sha256(Path(controlled.__file__)),
              "seeds": list(SEEDS), "input_columns": numeric, "categorical_columns": [],
              "removed_columns": ["dataset_group", "chemistry_system"], "ridge": {"alpha": 1000.0},
              "xgboost": controlled.XGB_PARAMETERS, "ridge_execution": "two deterministic fits, outputs copied to 23 seed indices",
              "training_data_reads": [relative(p) for p in paths], "test_labels_read_during_fit": False,
              "inputs_sha256": {relative(p): analysis.sha256(p) for p in paths},
              "command": "python 02_experiments/analysis/run_metadata_ablation_v14.py fit --data-root DATA --output FIT_RESULTS"}
    write_json(OUT / "frozen_configuration.json", config)
    train = features[features.split.eq("train")]
    output = []
    for model in controlled.CLASSICAL:
        actual_seeds = (SEEDS[0],) if model == "ridge" else SEEDS
        for seed in actual_seeds:
            for contract in controlled.CONTRACTS:
                started = time.perf_counter()
                record = {"model": model, "seed": int(seed), "fit_contract": contract, "started_utc": now()}
                try:
                    estimator = controlled.make_classical(model, seed, numeric, [])
                    estimator.fit(train[numeric], train.file_name.map(labels[contract]).to_numpy(float))
                    assert list(estimator.feature_names_in_) == numeric
                    assert list(estimator.named_steps["preprocess"].get_feature_names_out()) == ["numeric__" + n for n in numeric]
                    raw_rows = []
                    for split in ("calibration", "test"):
                        subset = features[features.split.eq(split)]
                        predicted = estimator.predict(subset[numeric])
                        assert np.isfinite(predicted).all()
                        for row, value in zip(subset.itertuples(), predicted):
                            raw_rows.append({"fit_contract": contract, "split": split, "dataset_group": row.dataset_group,
                                             "chemistry_system": row.chemistry_system, "file_name": row.file_name,
                                             "model": model, "seed": int(seed), "predicted_eol": max(float(value), 1.0),
                                             "prediction_kind": "controlled_early_only_blind"})
                    copies = SEEDS if model == "ridge" else (seed,)
                    output.extend({**row, "seed": int(index)} for index in copies for row in raw_rows)
                    record.update(status="completed", seconds=time.perf_counter() - started,
                                  fitted_columns=list(estimator.feature_names_in_), actual_fitted_rows=503,
                                  prediction_rows_before_seed_copy=320, represented_seed_indices=list(copies),
                                  state_sha256=controlled.object_sha256(estimator),
                                  target_sha256=controlled.object_sha256(train.file_name.map(labels[contract]).to_numpy(float)))
                    controlled.append_jsonl(OUT / "training_runs.jsonl", record)
                    print(f"completed {model} {seed} {contract}: {record['seconds']:.3f}s", flush=True)
                except Exception as exc:
                    record.update(status="failed", seconds=time.perf_counter() - started, error=repr(exc))
                    controlled.append_jsonl(OUT / "training_runs.jsonl", record)
                    raise
    prediction_path = OUT / "classical_blind_predictions.csv"
    pd.DataFrame(output).to_csv(prediction_path, index=False)
    write_json(OUT / "training_manifest.json", {"status": "COMPLETE_PENDING_INDEPENDENT_VALIDATION", "finished_utc": now(),
               "actual_fits": 48, "represented_seed_contract_model_cells": 92, "rows": len(output),
               "predictions_sha256": analysis.sha256(prediction_path), "log_sha256": analysis.sha256(OUT / "training_runs.jsonl"),
               "configuration_sha256": analysis.sha256(OUT / "frozen_configuration.json"), **resource_receipt(start)})


def analyze():
    start = time.perf_counter()
    blind = pd.read_csv(FIT_RESULTS / "classical_blind_predictions.csv")
    retained = pd.read_csv(CORE / "blind_predictions.csv")
    deep = retained[retained.model.isin(controlled.DEEP)]
    combined = pd.concat([blind, deep], ignore_index=True)
    combined.to_csv(OUT / "combined_blind_predictions.csv", index=False)
    labels = analysis.load_labels(INPUT / "labels")
    paired_path = ROOT / "03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv"
    paired = analysis.paired_metadata(paired_path)
    revealed, points, factors = analysis.point_outputs(combined, labels)
    ranks, comparisons, best, pairwise = analysis.ranking_outputs(points)
    wide, cross, within, variation = analysis.refit_variability(combined, paired)
    print("Analyzing 5,000 crossed seed/cell bootstrap replicates with the original definitions", flush=True)
    replicates, summary = analysis.prediction_shift_bootstrap(wide, 5000)
    outputs = {"point_metrics_by_seed.csv": points, "factor_contrasts_by_seed.csv": factors,
               "model_ranks_by_cell.csv": ranks, "ranking_comparisons.csv": comparisons, "best_model_frequency.csv": best,
               "pairwise_model_wins.csv": pairwise, "prediction_shift_cells.csv": wide,
               "cross_contract_prediction_shift_by_seed.csv": cross, "within_contract_refit_pairwise.csv": within,
               "refit_variability_summary.csv": variation, "prediction_shift_bootstrap_replicates.csv": replicates,
               "prediction_shift_bootstrap_summary.csv": summary}
    reference = pd.read_csv(BASE / "prediction_shift_bootstrap_summary.csv")
    cols = ["model", "label_status", "bootstrap_median", "bootstrap_95_low", "bootstrap_95_high"]
    compare = summary[summary.model.eq("xgboost") & summary.quantity.eq("cross_to_within_ratio")][cols].merge(
        reference[reference.model.eq("xgboost") & reference.quantity.eq("cross_to_within_ratio")][cols],
        on=["model", "label_status"], suffixes=("_early_only", "_metadata_aware"))
    outputs["xgboost_ratio_comparison.csv"] = compare
    matched = comparisons[comparisons.direction.eq("matched_contract")]
    result = {"actual_new_fits": 48, "xgboost_fits": 46, "ridge_fits": 2, "reused_deep_prediction_rows": len(deep),
              "combined_prediction_rows": len(combined), "xgboost_rank1_opportunities": int(best.loc[best.model.eq("xgboost"), "best_seed_count"].sum()),
              "rank1_total_opportunities": 230, "winner_changes": int(matched.best_model_changed.sum()),
              "matched_comparisons": len(matched), "full_order_same": int(matched.full_order_same.sum()),
              "top2_same": int(matched.top2_set_same.sum()), "ratio_comparison": compare.to_dict("records"),
              "inference_boundary": "explicit indicator removal; curves can still encode source identity; no independent dataset or configuration search"}
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    write_json(OUT / "result_summary.json", result)
    input_paths = [Path(__file__), Path(analysis.__file__), CORE / "blind_predictions.csv", paired_path,
                   FIT_RESULTS / "classical_blind_predictions.csv", BASE / "prediction_shift_bootstrap_summary.csv"]
    write_json(OUT / "analysis_manifest.json", {"status": "COMPLETE_PENDING_INDEPENDENT_VALIDATION", "finished_utc": now(),
               "command": "python 02_experiments/analysis/run_metadata_ablation_v14.py analyze --data-root DATA --fit-results FIT_RESULTS --output ANALYSIS", "bootstrap_reps": 5000,
               "inputs_sha256": {relative(p): analysis.sha256(p) for p in input_paths},
               "outputs_sha256": {name: analysis.sha256(OUT / name) for name in [*outputs, "result_summary.json", "combined_blind_predictions.csv"]},
               **resource_receipt(start)})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit or analyze models without explicit source and chemistry indicators.")
    parser.add_argument("stage", choices=("fit", "analyze"))
    parser.add_argument("--data-root", type=Path, required=True, help="Root of the extracted derived-data package")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--fit-results", type=Path, help="Fitted classical predictions; defaults to the stored data-package run")
    args = parser.parse_args()
    ROOT = args.data_root.resolve()
    OUT = args.output.resolve()
    FIT_RESULTS = (args.fit_results or ROOT / "04_analysis/reviewer_v14_20260905/metadata_ablation").resolve()
    INPUT = ROOT / "03_data_interaction/model_input/gate1_v2_20260821_0656"
    CORE = ROOT / "04_analysis/reviewer_v13_20260904/core"
    BASE = CORE.parent / "analysis"
    if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
        raise SystemExit("Output must be a new or empty directory")
    if args.stage == "fit" and args.fit_results:
        parser.error("--fit-results applies only to analyze")
    OUT.mkdir(parents=True, exist_ok=True)
    fit() if args.stage == "fit" else analyze()
