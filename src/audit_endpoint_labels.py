"""Recalculate endpoint labels from processed BatteryLife cell objects."""

import argparse
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def nominal_capacity(name, cell):
    if name.startswith("RWTH"):
        return 1.85
    if name.startswith("SNL_18650_NCA_25C_20-80"):
        return 3.2
    return float(cell["nominal_capacity_in_Ah"])


def standard_label(name, cell):
    nominal = nominal_capacity(name, cell)
    span = float(cell["SOC_interval"][1]) - float(cell["SOC_interval"][0])
    if span == 0:
        span = 1.0
    cycles = cell["cycle_data"]
    final_soh = max(cycles[-1]["discharge_capacity_in_Ah"]) / nominal / span
    if final_soh >= 0.825:
        return np.nan, "excluded_final_soh_ge_0.825", final_soh
    if final_soh > 0.8:
        values = np.asarray([max(cycle["discharge_capacity_in_Ah"]) / nominal / span for cycle in cycles[-20:]]).reshape(-1, 1)
        numbers = np.arange(len(cycles) - 19, len(cycles) + 1)
        model = LinearRegression().fit(values, numbers)
        return int(model.predict([[0.8]])[0]), "extrapolated_final20", final_soh
    for index, cycle in enumerate(cycles, 1):
        soh = max(cycle["discharge_capacity_in_Ah"]) / nominal / span
        if soh <= 0.8:
            return index, "observed_first_crossing", final_soh
    return np.nan, "unresolved", final_soh


def xjtu_label(cell, tools):
    nominal = float(cell["nominal_capacity_in_Ah"])
    span = tools.get_soc_span(cell)
    numbers, values = tools.collect_cycle_soh_curve(cell["cycle_data"], nominal, span)
    label = tools._linear_interpolate_last_descending_x_at_y(numbers, values, 0.8)
    kind = "observed_interpolated_crossing"
    finite = values[np.isfinite(values)]
    final_soh = float(finite[-1]) if len(finite) else np.nan
    if not np.isfinite(label) and np.isfinite(final_soh) and 0.8 < final_soh < 0.825:
        label = tools._linear_extrapolate_x_at_y_from_tail(numbers, values, 0.8)
        kind = "extrapolated_final20"
    if not np.isfinite(label):
        kind = "unresolved"
    return tools.cycle_life_label_to_int(label), kind, final_soh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "process_scripts"))
    from Extract_life_labels_tools import XJTU_tools
    taxonomy = pd.read_csv(args.taxonomy)
    rows = []
    for row in taxonomy.itertuples():
        path = args.processed / row.dataset_group / row.file_name
        if not path.is_file():
            rows.append({"split": row.split, "dataset_group": row.dataset_group, "file_name": row.file_name, "status": row.status, "released_eol": row.eol_cycle, "recomputed_eol": np.nan, "endpoint_kind": "missing_file", "final_soh": np.nan, "label_relation": "missing_file", "strict_label_match": False, "endpoint_consistent": False})
            continue
        with path.open("rb") as handle:
            cell = pickle.load(handle)
        recomputed, kind, final_soh = xjtu_label(cell, XJTU_tools) if row.dataset_group == "XJTU" else standard_label(path.name, cell)
        released = float(row.eol_cycle) if pd.notna(row.eol_cycle) else np.nan
        strict_match = bool((not np.isfinite(released) and not np.isfinite(recomputed)) or (np.isfinite(released) and np.isfinite(recomputed) and released == recomputed))
        if strict_match:
            relation = "exact"
        elif row.dataset_group == "Tongji" and np.isfinite(released) and np.isfinite(recomputed) and released - recomputed == 1:
            relation = "tongji_released_plus_one"
        else:
            relation = "unexplained_mismatch"
        rows.append({
            "split": row.split,
            "dataset_group": row.dataset_group,
            "file_name": row.file_name,
            "status": row.status,
            "released_eol": released,
            "recomputed_eol": recomputed,
            "endpoint_kind": kind,
            "final_soh": final_soh,
            "label_relation": relation,
            "strict_label_match": strict_match,
            "endpoint_consistent": relation in ["exact", "tongji_released_plus_one"],
        })
    frame = pd.DataFrame(rows)
    expected = frame[frame["status"].isin(["usable", "eol_not_after_observation"])]
    mismatches = expected[~expected["endpoint_consistent"]]
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": "source-generated 80% normalized-capacity cycle life",
        "listed_cells": len(frame),
        "released_label_cells": len(expected),
        "endpoint_kind_counts": dict(Counter(expected["endpoint_kind"])),
        "label_relation_counts": dict(Counter(expected["label_relation"])),
        "strict_match_count": int(expected["strict_label_match"].sum()),
        "recognized_tongji_plus_one_count": int((expected["label_relation"] == "tongji_released_plus_one").sum()),
        "unexplained_mismatch_count": len(mismatches),
        "mismatches": mismatches[["split", "dataset_group", "file_name", "released_eol", "recomputed_eol", "endpoint_kind", "label_relation"]].to_dict("records"),
        "pass": len(mismatches) == 0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "endpoint_label_cells.csv", index=False)
    (args.out / "endpoint_label_audit.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    raise SystemExit(0 if record["pass"] else 1)


if __name__ == "__main__":
    main()
