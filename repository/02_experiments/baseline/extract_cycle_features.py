import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CHECKPOINTS = (1, 2, 5, 10, 20, 50, 75, 100)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interp(values, length=64):
    source = np.arange(1, len(values) + 1)
    target = np.linspace(1, len(values) + 1, num=length, endpoint=True)
    return np.interp(target, source, values)


def resample_cycle(cycle, nominal_capacity, discharge_first, curve_length=300):
    keys = ["current_in_A", "voltage_in_V", "charge_capacity_in_Ah", "discharge_capacity_in_Ah", "time_in_s"]
    frame = pd.DataFrame({key: cycle[key] for key in keys})
    invalid = (frame["charge_capacity_in_Ah"] < 0) | (frame["discharge_capacity_in_Ah"] < 0)
    frame.loc[invalid] = np.nan
    frame.bfill(inplace=True)
    current = frame["current_in_A"].to_numpy(float)
    voltage = frame["voltage_in_V"].to_numpy(float)
    charge_capacity = frame["charge_capacity_in_Ah"].to_numpy(float)
    discharge_capacity = frame["discharge_capacity_in_Ah"].to_numpy(float)
    current_c = current / nominal_capacity
    charge_end = np.flatnonzero(current_c >= 0.01)[-1]
    discharge_end = np.flatnonzero(current_c <= -0.01)[-1]
    if discharge_first:
        dv, di, dq = voltage[:discharge_end], current[:discharge_end], discharge_capacity[:discharge_end]
        cv, ci, cq = voltage[discharge_end:], current[discharge_end:], charge_capacity[discharge_end:]
        keep = np.abs(ci / nominal_capacity) > 0.01
        cv, ci, cq = cv[keep], ci[keep], cq[keep]
    else:
        dv, di, dq = voltage[charge_end:], current[charge_end:], discharge_capacity[charge_end:]
        keep = np.abs(di / nominal_capacity) > 0.01
        dv, di, dq = dv[keep], di[keep], dq[keep]
        cv, ci, cq = voltage[:charge_end], current[:charge_end], charge_capacity[:charge_end]
    half = curve_length // 2
    v = np.concatenate([interp(cv, half), interp(dv, half)])
    i = np.concatenate([interp(ci, half), interp(di, half)]) / nominal_capacity
    q = np.concatenate([interp(cq, half), interp(dq, half)]) / nominal_capacity
    return np.stack([v / np.nanmax(v), i, q]).astype(np.float32)


def summary_features(sequence):
    half = sequence.shape[-1] // 2
    series = {
        "charge_capacity": np.nanmax(sequence[:, 2, :half], axis=1),
        "discharge_capacity": np.nanmax(sequence[:, 2, half:], axis=1),
        "coulombic_efficiency": np.nanmax(sequence[:, 2, half:], axis=1) / np.maximum(np.nanmax(sequence[:, 2, :half], axis=1), 1e-6),
        "charge_voltage": np.nanmean(sequence[:, 0, :half], axis=1),
        "discharge_voltage": np.nanmean(sequence[:, 0, half:], axis=1),
        "charge_c_rate": np.nanmedian(np.abs(sequence[:, 1, :half]), axis=1),
        "discharge_c_rate": np.nanmedian(np.abs(sequence[:, 1, half:]), axis=1),
    }
    features = {}
    x = np.arange(1, sequence.shape[0] + 1)
    for name, values in series.items():
        for cycle in CHECKPOINTS:
            if cycle <= len(values):
                features[f"{name}_c{cycle}"] = values[cycle - 1]
        for left, right in ((1, len(values)), (10, len(values)), (50, len(values))):
            if left <= right:
                window = values[left - 1:right]
                features[f"{name}_mean_{left}_{right}"] = np.mean(window)
                features[f"{name}_std_{left}_{right}"] = np.std(window)
                features[f"{name}_slope_{left}_{right}"] = np.polyfit(x[left - 1:right], window, 1)[0]
        features[f"{name}_delta_10_{len(values)}"] = values[-1] - values[min(9, len(values) - 1)]
    return features


def main():
    parser = argparse.ArgumentParser(description="Extract early-cycle inputs from trusted BatteryLife processed objects and the supplied cohort tables.")
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--features-out", type=Path, required=True)
    parser.add_argument("--sequences-out", type=Path, required=True)
    parser.add_argument("--paired-labels", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()
    outputs = [args.features_out, args.sequences_out, args.manifest_out]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise SystemExit("Feature, sequence and manifest outputs must be distinct new files")
    if args.labels_root.exists() and any(args.labels_root.iterdir()):
        raise SystemExit("Label output directory must be new or empty")
    taxonomy = pd.read_csv(args.taxonomy)
    required = {"split", "dataset_group", "chemistry_system", "file_name"}
    if not required.issubset(taxonomy.columns) or not {"analysis_status", "status"}.intersection(taxonomy.columns):
        raise SystemExit("Taxonomy needs split, dataset_group, chemistry_system, file_name and status (usable) or analysis_status (included)")
    selection_column = "analysis_status" if "analysis_status" in taxonomy else "status"
    selection_value = "included" if selection_column == "analysis_status" else "usable"
    taxonomy = taxonomy[taxonomy[selection_column] == selection_value].reset_index(drop=True)
    paired = pd.read_csv(args.paired_labels)
    if not {"split", "file_name", "release_eol", "current_generator_eol"}.issubset(paired.columns):
        raise SystemExit("Paired labels need split, file_name, release_eol and current_generator_eol")
    if taxonomy.file_name.duplicated().any() or paired.file_name.duplicated().any():
        raise SystemExit("Cell identifiers must be unique")
    if len(paired) != len(taxonomy) or set(paired["file_name"]) != set(taxonomy["file_name"]):
        raise SystemExit("Taxonomy and paired label contract identifiers differ")
    expected_splits = taxonomy.set_index("file_name")["split"].to_dict()
    if any(expected_splits[row.file_name] != row.split for row in paired.itertuples()):
        raise SystemExit("Taxonomy and paired label contract splits differ")
    missing = [str(args.processed / row.dataset_group / row.file_name) for row in taxonomy.itertuples() if not (args.processed / row.dataset_group / row.file_name).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} usable curve files; first: {missing[0]}")
    rows, sequences = [], []
    for row in taxonomy.itertuples():
        path = args.processed / row.dataset_group / row.file_name
        with path.open("rb") as handle:
            cell = pickle.load(handle)
        if len(cell["cycle_data"]) < args.horizon:
            raise RuntimeError(f"Fewer than {args.horizon} cycles: {path}")
        if row.file_name.startswith("RWTH"):
            nominal = 1.85
        elif row.file_name.startswith("SNL_18650_NCA_25C_20-80"):
            nominal = 3.2
        else:
            nominal = float(cell["nominal_capacity_in_Ah"])
        sequence = np.stack([
            resample_cycle(cycle, nominal, row.file_name.startswith("RWTH"))
            for cycle in cell["cycle_data"][:args.horizon]
        ])
        if not np.isfinite(sequence).all():
            raise RuntimeError(f"Non-finite CyclePatch input: {path}")
        record = {
            "split": row.split,
            "dataset_group": row.dataset_group,
            "chemistry_system": row.chemistry_system,
            "file_name": row.file_name,
            **summary_features(sequence),
        }
        rows.append(record)
        sequences.append(sequence)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame = pd.DataFrame(rows)
    numeric = feature_frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError("Non-finite summary feature")
    feature_frame.to_csv(args.features_out, index=False)
    curve_array = np.stack(sequences)
    np.savez_compressed(
        args.sequences_out,
        curves=curve_array,
        split=taxonomy["split"].to_numpy(),
        dataset_group=taxonomy["dataset_group"].to_numpy(),
        chemistry_system=taxonomy["chemistry_system"].to_numpy(),
        file_name=taxonomy["file_name"].to_numpy(),
    )
    label_paths = {}
    for contract, column in (("release", "release_eol"), ("current_generator", "current_generator_eol")):
        label_dir = args.labels_root / contract
        label_dir.mkdir(parents=True, exist_ok=True)
        for split, file_name in (("train", "train_labels.csv"), ("calibration", "calibration_labels.csv"), ("test", "test_labels_sealed.csv")):
            label_path = label_dir / file_name
            frame = paired.loc[paired["split"] == split, ["file_name", column]].rename(columns={column: "eol_cycle"})
            frame.to_csv(label_path, index=False)
            label_paths[f"{contract}:{split}"] = label_path
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "horizon": args.horizon,
        "cells": len(rows),
        "sequence_shape": list(curve_array.shape),
        "features_sha256": sha256(args.features_out),
        "sequences_sha256": sha256(args.sequences_out),
        "paired_labels_sha256": sha256(args.paired_labels),
        "taxonomy_sha256": sha256(args.taxonomy),
        "source_commit": "3b17bb53202e10711deae0448c91f84144bb3dd9",
        "processed_source": "Zenodo 21149533",
        "label_sha256": {split: sha256(path) for split, path in label_paths.items()},
        "future_cycles_read": False,
        "resampling": "BatteryLife data_loader.py logic, 150 points per charge/discharge segment",
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
