import csv
import argparse
import hashlib
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path


parser = argparse.ArgumentParser(description="Check the supplied Tongji trace, or reconstruct it from the external processed object.")
parser.add_argument("--data-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--raw-reconstruct", action="store_true")
parser.add_argument("--processed-pickle", type=Path)
parser.add_argument("--archived-labels", type=Path)
parser.add_argument("--battery-life-root", type=Path)
args = parser.parse_args()
ROOT = args.data_root.resolve()
OUT = args.output.resolve()
if OUT.is_relative_to(ROOT) or (OUT.exists() and any(OUT.iterdir())):
    parser.error("--output must be a new or empty directory outside --data-root")
CELL = "Tongji1_CY25-025_1--7.pkl"
COMMIT = "3b17bb53202e10711deae0448c91f84144bb3dd9"
PICKLE_SHA = "6a2cf73535b2b1069602253b0b1ab73abea295c68772d3f1edc2597850146125"
INPUTS = {
    "paired_labels": "03_data_interaction/data_integrity/paired_label_contracts/paired_label_contracts.csv",
    "endpoint_audit": "03_data_interaction/data_integrity/endpoint_audit/endpoint_label_cells.csv",
    "curve_manifest": "03_data_interaction/data_integrity/curve_audit/processed_curve_cells.csv",
    "archived_labels": "01_raw_data/zenodo_21149533/processed/Life labels/Tongji_labels.json",
    "exception_pickle": f"01_raw_data/zenodo_21149533/processed/Tongji/{CELL}",
}
UPSTREAM = {
    "process_scripts/preprocess_Tongji.py": "8ea751a2eaa7d5c78e3b73e620616af48c54d5bde513e55146286f82fd02d5d7",
    "process_scripts/Extract_life_labels.py": "70491f065757a15ffe364792ade2b300ad0f6ab2f42e05b57e89f053b589c92c",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(key):
    with (ROOT / INPUTS[key]).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


paired = [row for row in rows("paired_labels") if row["dataset_group"] == "Tongji"]
audited = {row["file_name"]: row for row in rows("endpoint_audit")}
manifest = {row["file_name"]: row for row in rows("curve_manifest")}
changed = [row for row in paired if float(row["label_delta_cycles"]) != 0]
unchanged = [row for row in paired if float(row["label_delta_cycles"]) == 0]
branches = Counter(audited[row["file_name"]]["endpoint_kind"] for row in changed)
if not args.raw_reconstruct:
    evidence = ROOT / "04_analysis/reviewer_v14_20260905/tongji_trace"
    record = json.loads((evidence / "audit.json").read_text(encoding="utf-8"))
    with (evidence / "exception_trace.csv").open(encoding="utf-8", newline="") as handle:
        trace = list(csv.DictReader(handle))
    positions = {int(row["retained_position"]): row for row in trace}
    checks = {
        "trace_hash_matches_stored_receipt": digest(evidence / "exception_trace.csv") == record["trace_sha256"],
        "cohort_inputs_match_stored_receipt": all(digest(ROOT / INPUTS[key]) == record["inputs"][key]["sha256"] for key in ["paired_labels", "endpoint_audit", "curve_manifest"]),
        "cohort_99_changed_98_unchanged_1": (len(paired), len(changed), len(unchanged)) == (99, 98, 1),
        "all_changed_deltas_minus_one": all(float(row["label_delta_cycles"]) == -1 for row in changed),
        "unique_unchanged_identity_labels_and_split": len(unchanged) == 1 and unchanged[0]["file_name"] == CELL and unchanged[0]["split"] == "test" and float(unchanged[0]["release_eol"]) == float(unchanged[0]["current_generator_eol"]) == 201,
        "changed_branches_90_observed_8_extrapolated": branches == {"observed_first_crossing": 90, "extrapolated_final20": 8},
        "all_endpoint_audit_labels_match_paired": all(float(audited[row["file_name"]]["released_eol"]) == float(row["release_eol"]) and float(audited[row["file_name"]]["recomputed_eol"]) == float(row["current_generator_eol"]) for row in paired),
        "five_supplied_positions_199_to_203": len(trace) == 5 and set(positions) == set(range(199, 204)) and all(row["file_name"] == CELL for row in trace),
        "local_strict_crossing_bracket_200_201": float(positions[200]["normalised_capacity"]) > .8 > float(positions[201]["normalised_capacity"]),
        "local_reference_and_raw_counter": all(abs(float(row["max_discharge_capacity_Ah"]) / 3.5 - float(row["normalised_capacity"])) < 1e-12 and abs(float(row["threshold_Ah"]) - 2.8) < 1e-12 for row in trace) and int(positions[201]["stored_cycle_number"]) == 210,
        "frozen_object_digest_matches_manifest": record["inputs"]["exception_pickle"]["sha256"] == PICKLE_SHA == manifest[CELL]["file_sha256"],
        "historical_cause_not_recovered": record["historical_cause_recovered"] is False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "mode": "stored_trace_validation", "checks": checks, "scope": "Checks the supplied five-position trace and derived cohort tables. It does not reread the processed object, establish first crossing over all 251 positions, or rerun the upstream generator.", "input_sha256": {name: digest(evidence / name) for name in ["audit.json", "exception_trace.csv"]}}
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "mode": result["mode"], "passed": sum(checks.values()), "total": len(checks)}))
    raise SystemExit(0 if all(checks.values()) else 1)
if not all([args.processed_pickle, args.archived_labels, args.battery_life_root]):
    parser.error("--raw-reconstruct requires --processed-pickle, --archived-labels and --battery-life-root; only load a trusted pickle")
INPUTS["archived_labels"] = args.archived_labels.resolve()
INPUTS["exception_pickle"] = args.processed_pickle.resolve()
archived = json.loads((ROOT / INPUTS["archived_labels"]).read_text(encoding="utf-8"))
with (ROOT / INPUTS["exception_pickle"]).open("rb") as handle:
    cell = pickle.load(handle)
cycles = cell["cycle_data"]
nominal = float(cell["nominal_capacity_in_Ah"])
span = float(cell["SOC_interval"][1] - cell["SOC_interval"][0]) or 1.0
capacity = [float(max(cycle["discharge_capacity_in_Ah"])) for cycle in cycles]
soh = [value / nominal / span for value in capacity]
crossing = next(index for index, value in enumerate(soh, 1) if value <= 0.8)
trace = [{"file_name": CELL, "retained_position": i + 1,
          "stored_cycle_number": cycles[i]["cycle_number"], "max_discharge_capacity_Ah": capacity[i],
          "normalised_capacity": soh[i], "threshold_Ah": 0.8 * nominal * span,
          "at_or_below_threshold": soh[i] <= 0.8,
          "is_first_crossing": i + 1 == crossing} for i in range(crossing - 3, crossing + 2)]
repo = args.battery_life_root.resolve()
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--", *UPSTREAM], text=True).strip()
checks = {
    "cohort_99_changed_98_unchanged_1": (len(paired), len(changed), len(unchanged)) == (99, 98, 1),
    "all_changed_deltas_minus_one": all(float(row["label_delta_cycles"]) == -1 for row in changed),
    "unique_unchanged_identity_labels_and_split": len(unchanged) == 1 and unchanged[0]["file_name"] == CELL
        and unchanged[0]["split"] == "test" and float(unchanged[0]["release_eol"]) == float(unchanged[0]["current_generator_eol"]) == 201,
    "changed_branches_90_observed_8_extrapolated": branches == {"observed_first_crossing": 90, "extrapolated_final20": 8},
    "all_archived_labels_match_paired": all(float(archived[row["file_name"].replace("--", "-#")]) == float(row["release_eol"]) for row in paired),
    "all_endpoint_audit_labels_match_paired": all(float(audited[row["file_name"]]["released_eol"]) == float(row["release_eol"])
        and float(audited[row["file_name"]]["recomputed_eol"]) == float(row["current_generator_eol"]) for row in paired),
    "exception_is_observed_first_crossing_201": crossing == 201 and soh[-1] < 0.8
        and audited[CELL]["endpoint_kind"] == "observed_first_crossing",
    "exception_strict_threshold_bracket": soh[199] > 0.8 > soh[200] and all(value > 0.8 for value in soh[:200]),
    "exception_coordinate_and_reference": len(cycles) == 251 and nominal == 3.5 and span == 1
        and cycles[0]["cycle_number"] == 2 and cycles[200]["cycle_number"] == 210,
    "exception_pickle_matches_frozen_hash_and_manifest": digest(ROOT / INPUTS["exception_pickle"]) == PICKLE_SHA == manifest[CELL]["file_sha256"],
    "fixed_upstream_commit": head == COMMIT,
    "upstream_source_clean_and_hashes_match": not dirty and all(digest(repo / path) == sha for path, sha in UPSTREAM.items()),
}
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "exception_trace.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(trace[0]))
    writer.writeheader()
    writer.writerows(trace)
record = {
    "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
    "cohort_cells": len(paired), "changed_cells": len(changed), "unchanged_cells": len(unchanged),
    "changed_current_generator_branches": dict(sorted(branches.items())),
    "exception": {"file_name": CELL, "split": "test", "contract_a": 201, "contract_b": crossing,
        "retained_cycles": len(cycles), "nominal_capacity_Ah": nominal, "soc_span": span,
        "final_normalised_capacity": soh[-1], "current_branch": audited[CELL]["endpoint_kind"]},
    "historical_cause_recovered": False,
    "interpretation": "Near-uniform one-position numerical pattern consistent with an indexing convention; historical archived-label generation and the cause of its exception remain unrecovered.",
    "boundary": "The exception is an ordinary strict first crossing, not an equality, rounding, or terminal-extrapolation boundary. Stored raw cycle numbers and retained positions are distinct.",
    "mode": "raw_reconstruction",
    "inputs": {name: {"path": Path(path).name if Path(path).is_absolute() else str(path), "sha256": digest(ROOT / path)} for name, path in INPUTS.items()},
    "upstream": {"commit": head, "sources": UPSTREAM, "sources_clean": not dirty},
    "script_sha256": digest(Path(__file__)), "trace_sha256": digest(OUT / "exception_trace.csv"),
}
(OUT / "audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
(OUT / "README.md").write_text("""# Tongji one-position discrepancy and unchanged-cell trace

Reconstruct with `--data-root DATA --output RESULTS/tongji_raw --raw-reconstruct --processed-pickle PATH --archived-labels PATH --battery-life-root PATH`. Load only trusted processed objects. Without `--raw-reconstruct`, the script checks the supplied derived trace and cohort tables; that offline check does not reread the original trajectory.

The frozen model cohort contains 99 Tongji cells: 98 have B-A=-1 and one has no change. The changed cells comprise 90 current observed-first-crossing labels and 8 current final-20 extrapolated labels. The sole unchanged cell, `Tongji1_CY25-025_1--7.pkl` in the test split, has A=B=201.

The script checks all 99 paired labels against the archived JSON and the existing endpoint audit, then independently reads only the unchanged cell's hash-matched processed object. Its current generator branch is observed first crossing: normalised capacity falls from 0.8005469853 at retained position 200 to 0.7992457643 at position 201. The final value is 0.7163325223, so terminal extrapolation is not used. This is not an exact-threshold or rounding-boundary case. Retained position 201 carries stored cycle number 210; the two coordinates must not be conflated.

The near-uniform numerical difference is consistent with an indexing convention. It does not establish the historical cause of the archived-label offset or its exception. The pinned preprocessor omits raw cycles below 2 and also filters cycles by capacity; those code facts do not identify the unrecovered historical A generator.

`exception_trace.csv` provides the five local positions around the crossing. `audit.json` records checks, relative input paths, SHA-256 hashes and the pinned commit. The raw processed pickle is an external input and is not redistributed in this evidence directory. This check does not rerun model fitting or recompute the other 98 raw trajectories; their branch counts are reconciled against the frozen endpoint audit.
""", encoding="utf-8")
print(json.dumps({"status": record["status"], "passed_checks": sum(checks.values()), "total_checks": len(checks)}))
raise SystemExit(0 if all(checks.values()) else 1)
