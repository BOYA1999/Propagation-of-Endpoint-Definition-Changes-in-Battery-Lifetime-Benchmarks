import csv
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Validate stored ISU coordinate evidence or reconstruct it from external processed objects.")
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--raw-reconstruct", action="store_true")
parser.add_argument("--processed-dir", type=Path)
parser.add_argument("--archived-labels", type=Path)
args = parser.parse_args()
ROOT = args.data_root.resolve()
LINEAGE_PATH = ROOT / "03_data_interaction/data_integrity/isu_endpoint_lineage/isu_endpoint_lineage_cells.csv"
PAIRED_PATH = ROOT / "03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv"
OUT_DIR = args.output.resolve()
if OUT_DIR.is_relative_to(ROOT) or (OUT_DIR.exists() and any(OUT_DIR.iterdir())):
    parser.error("--output must be a new or empty directory outside --data-root")


def endpoint(q, reference):
    soh = q / reference
    if soh[-1] >= 0.825:
        return None, "excluded", float(soh[-1])
    if soh[-1] > 0.8:
        cycle_numbers = np.arange(len(q) - 19, len(q) + 1)
        value = LinearRegression().fit(soh[-20:, None], cycle_numbers).predict([[0.8]])[0]
        return int(value), "final20", float(soh[-1])
    return int(np.flatnonzero(soh <= 0.8)[0] + 1), "first", float(soh[-1])


def archived_effective_endpoint(q, reference):
    soh = q / reference
    crossings = np.flatnonzero(soh <= 0.8)
    if len(crossings):
        return int(crossings[0] + 1), "first"
    return len(q) + 1, "n_plus_one"


if not args.raw_reconstruct:
    evidence = ROOT / "03_data_interaction/data_integrity/isu_contract_coordinate_audit"
    summary = json.loads((evidence / "contract_coordinate_summary.json").read_text(encoding="utf-8"))
    with (evidence / "contract_coordinate_cells.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with PAIRED_PATH.open(encoding="utf-8", newline="") as handle:
        paired = [row for row in csv.DictReader(handle) if row["dataset_group"] == "ISU_ILCC"]
    by_name = {row["file_name"]: row for row in rows}
    g34 = by_name["ISU-ILCC_G34C2.pkl"]
    checks = {
        "240_unique_records": len(rows) == len(by_name) == summary["n_cells"] == 240,
        "archived_a_240_exact_stored_reconstructions": all(float(row["archived_a"]) == float(row["reconstructed_a"]) and row["archive_a_match"] == "True" for row in rows) and summary["archived_a_exact_matches"] == 240,
        "current_b_240_exact_stored_reconstructions": all(float(row["current_b"]) == float(row["lineage_b"]) and row["current_b_match"] == "True" for row in rows) and summary["current_b_exact_matches"] == 240,
        "a_branches_239_first_one_n_plus_one": Counter(row["branch_a"] for row in rows) == {"first": 239, "n_plus_one": 1} and by_name["ISU-ILCC_G14C4.pkl"]["branch_a"] == "n_plus_one" and float(by_name["ISU-ILCC_G14C4.pkl"]["archived_a"]) == float(by_name["ISU-ILCC_G14C4.pkl"]["n_cycles"]) + 1,
        "final20_alternative_one_mismatch": [row["file_name"] for row in rows if row["current_branch_a_match"] != "True"] == ["ISU-ILCC_G14C4.pkl"] and summary["archived_a_matches_with_current_final20_branch"] == 239,
        "thresholds_match_stored_references": all(np.isclose(float(row["threshold_a_Ah"]), .8 * float(row["reference_a_Ah"]), rtol=0, atol=1e-14) and np.isclose(float(row["threshold_b_Ah"]), .8 * float(row["reference_b_Ah"]), rtol=0, atol=1e-14) for row in rows),
        "239_model_labels_reconciled": len(paired) == 239 and all(float(row["release_eol"]) == float(by_name[row["file_name"]]["archived_a"]) and float(row["current_generator_eol"]) == float(by_name[row["file_name"]]["current_b"]) for row in paired),
        "g34c2_coordinate_values": float(g34["archived_a"]) == float(g34["reconstructed_a"]) == 3109 and float(g34["current_b"]) == 317 and float(g34["n_cycles"]) == 16644,
        "shared_coordinate_and_separate_author_endpoint": summary["contract_a_b_processed_cycle_coordinate_aligned"] is True and summary["source_author_week_endpoint_used_as_contract_a"] is False and summary["author_weeks_vs_processed_cycles_aligned"] is False,
        "historical_a_generator_not_recovered": summary["historical_a_generator_recovered"] is False,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "mode": "stored_trace_validation", "checks": checks, "scope": "Reconciles the supplied coordinate records, summary and model labels. It does not reopen the 240 processed objects or independently reconstruct their trajectories.", "input_sha256": {name: hashlib.sha256((evidence / name).read_bytes()).hexdigest() for name in ["contract_coordinate_cells.csv", "contract_coordinate_summary.json"]}}
    (OUT_DIR / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "mode": result["mode"], "passed": sum(checks.values()), "total": len(checks)}))
    raise SystemExit(0 if all(checks.values()) else 1)
if not args.processed_dir or not args.archived_labels:
    parser.error("--raw-reconstruct requires --processed-dir and --archived-labels; only load trusted pickles")
from sklearn.linear_model import LinearRegression

PKL_DIR = args.processed_dir.resolve()
LABELS_PATH = args.archived_labels.resolve()
labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
with LINEAGE_PATH.open(encoding="utf-8", newline="") as handle:
    lineage_rows = list(csv.DictReader(handle))
lineage = {row["file_name"]: row for row in lineage_rows if row["file_name"]}
with PAIRED_PATH.open(encoding="utf-8", newline="") as handle:
    model_cohort = {
        row["file_name"]
        for row in csv.DictReader(handle)
        if row["dataset_group"] == "ISU_ILCC"
    }
author_feature_records = [
    row for row in lineage_rows if row["author_endpoint_exact"].lower() == "true"
]
author_feature_model_files = {
    row["file_name"] for row in author_feature_records if row["file_name"]
}

rows = []
for file_name, archived_a in sorted(labels.items()):
    with (PKL_DIR / file_name).open("rb") as handle:
        data = pickle.load(handle)
    q = np.array([max(cycle["discharge_capacity_in_Ah"]) for cycle in data["cycle_data"]], dtype=float)
    nominal = float(data["nominal_capacity_in_Ah"])
    reference_a = min(float(q[0]), nominal)
    span_b = float(data["SOC_interval"][1] - data["SOC_interval"][0]) or 1.0
    reference_b = nominal * span_b
    reconstructed_a, branch_a = archived_effective_endpoint(q, reference_a)
    current_branch_a, current_branch_a_kind, final_soh_a = endpoint(q, reference_a)
    current_b, branch_b, final_soh_b = endpoint(q, reference_b)
    expected_b = int(float(lineage[file_name]["recomputed_eol"]))
    rows.append(
        {
            "file_name": file_name,
            "archived_a": archived_a,
            "reconstructed_a": reconstructed_a,
            "archive_a_match": archived_a == reconstructed_a,
            "current_branch_a": current_branch_a,
            "current_branch_a_match": archived_a == current_branch_a,
            "current_b": current_b,
            "lineage_b": expected_b,
            "current_b_match": current_b == expected_b,
            "branch_a": branch_a,
            "current_branch_a_kind": current_branch_a_kind,
            "branch_b": branch_b,
            "n_cycles": len(q),
            "nominal_capacity_Ah": nominal,
            "reference_a_Ah": reference_a,
            "reference_b_Ah": reference_b,
            "threshold_a_Ah": 0.8 * reference_a,
            "threshold_b_Ah": 0.8 * reference_b,
            "final_soh_a": final_soh_a,
            "final_soh_b": final_soh_b,
        }
    )

OUT_DIR.mkdir(parents=True, exist_ok=True)
with (OUT_DIR / "contract_coordinate_cells.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

mismatches_a = [row for row in rows if not row["archive_a_match"]]
current_branch_mismatches_a = [row for row in rows if not row["current_branch_a_match"]]
mismatches_b = [row for row in rows if not row["current_b_match"]]
g34 = next(row for row in rows if row["file_name"] == "ISU-ILCC_G34C2.pkl")
summary = {
    "mode": "raw_reconstruction",
    "n_cells": len(rows),
    "archived_a_exact_matches": len(rows) - len(mismatches_a),
    "archived_a_mismatches": [row["file_name"] for row in mismatches_a],
    "archived_a_matches_with_current_final20_branch": len(rows) - len(current_branch_mismatches_a),
    "current_final20_branch_mismatches": [row["file_name"] for row in current_branch_mismatches_a],
    "current_b_exact_matches": len(rows) - len(mismatches_b),
    "current_b_mismatches": [row["file_name"] for row in mismatches_b],
    "a_branches": dict(Counter(row["branch_a"] for row in rows)),
    "b_branches": dict(Counter(row["branch_b"] for row in rows)),
    "labels_sha256": hashlib.sha256(LABELS_PATH.read_bytes()).hexdigest(),
    "inferred_a_effective_rule": "first crossing of 0.8 * min(nominal_capacity_Ah, first retained cycle maximum discharge capacity_Ah); N+1 if no crossing",
    "b_reference": "nominal_capacity_Ah * stored SOC span",
    "shared_output_coordinate": "one-based ordinal of the retained processed ordinary-cycle sequence",
    "author_weeks_vs_processed_cycles_aligned": False,
    "contract_a_b_processed_cycle_coordinate_aligned": True,
    "historical_a_generator_recovered": False,
    "effective_a_reconstruction_exact": "240/240",
    "author_feature_cells": len(author_feature_records),
    "model_cohort_cells": len(model_cohort),
    "author_feature_model_cohort_intersection": len(author_feature_model_files & model_cohort),
    "source_author_week_endpoint_used_as_contract_a": False,
    "g34c2": g34,
}
(OUT_DIR / "contract_coordinate_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)

assert len(rows) == 240
assert not mismatches_a
assert len(current_branch_mismatches_a) == 1 and current_branch_mismatches_a[0]["file_name"] == "ISU-ILCC_G14C4.pkl"
assert not mismatches_b
assert g34["archived_a"] == 3109 and g34["reconstructed_a"] == 3109 and g34["current_b"] == 317
print(json.dumps(summary, indent=2, ensure_ascii=False))
