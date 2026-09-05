import argparse
import hashlib
import json
import os
import pickle
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor


DEFAULT_SEEDS = tuple(range(20)) + (42, 2021, 2024)
CONTRACTS = ("release", "current_generator")
CLASSICAL = ("ridge", "xgboost")
DEEP = ("cpmlp", "cptransformer")
XGB_PARAMETERS = {
    "n_estimators": 700,
    "max_depth": 4,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}
DEEP_EPOCHS = {"cpmlp": 11, "cptransformer": 8}
BATTERYLIFE_COMMIT = "3b17bb53202e10711deae0448c91f84144bb3dd9"


def check_upstream(repo):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != BATTERYLIFE_COMMIT:
        raise SystemExit(f"BatteryLife must be checked out at {BATTERYLIFE_COMMIT}; found {commit}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("BatteryLife has modified tracked files; use a clean pinned checkout")
    if git("ls-files", "--others", "--exclude-standard", "models", "layers", "utils"):
        raise SystemExit("BatteryLife has untracked importable source files; use a clean pinned checkout")
    return commit


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def state_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def append_predictions(path, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def classical_pipeline(model, numeric, categorical):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    transformed = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preprocess", preprocess), ("model", transformed)])


def make_classical(name, seed, numeric, categorical):
    if name == "ridge":
        model = Ridge(alpha=1000.0)
    else:
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            **XGB_PARAMETERS,
        )
    return classical_pipeline(model, numeric, categorical)


def deep_config(name, horizon):
    shared = dict(charge_discharge_length=300, early_cycle_threshold=horizon, dropout=0.0)
    if name == "cpmlp":
        return SimpleNamespace(**shared, d_model=128, d_ff=256, e_layers=4, d_layers=2), 16
    return SimpleNamespace(
        **shared,
        d_model=128,
        d_ff=256,
        e_layers=6,
        d_layers=4,
        factor=3,
        n_heads=4,
        activation="gelu",
        output_num=1,
    ), 32


def deep_loader(curves, targets, indices, batch_size, shuffle, generator=None):
    x = torch.from_numpy(curves[indices]).float()
    mask = torch.ones((len(indices), curves.shape[1]), dtype=torch.float32)
    y = torch.from_numpy(targets[indices]).float().reshape(-1, 1)
    return DataLoader(
        TensorDataset(x, mask, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


def train_epoch(model, batches, optimizer, device):
    model.train()
    losses = []
    for curves, mask, target in batches:
        curves = curves.to(device)
        mask = mask.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(curves, mask)
        loss = torch.mean((prediction - target) ** 2)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    value = float(np.mean(losses))
    if not np.isfinite(value):
        raise RuntimeError("Non-finite deep-model loss")
    return value


def deep_predict(model, curves, indices, batch_size, device):
    batches = deep_loader(curves, np.zeros(len(curves), dtype=np.float32), indices, batch_size, False)
    output = []
    model.eval()
    with torch.inference_mode():
        for batch_curves, mask, _ in batches:
            output.append(model(batch_curves.to(device), mask.to(device)).cpu().numpy().reshape(-1))
    return np.concatenate(output)


def completed_runs(log_path):
    if not log_path.exists():
        return set()
    completed = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") == "completed":
            completed.add((record["model"], int(record["seed"]), record["fit_contract"]))
    return completed


def validate_labels(features, label_paths):
    train_ids = set(features.loc[features.split.eq("train"), "file_name"])
    labels = {}
    for contract, path in label_paths.items():
        frame = pd.read_csv(path)
        if len(frame) != len(train_ids) or set(frame.file_name) != train_ids or frame.file_name.duplicated().any():
            raise SystemExit(f"Invalid train-label alignment for {contract}")
        labels[contract] = frame.set_index("file_name").eol_cycle.astype(float).to_dict()
    return labels


def main():
    parser = argparse.ArgumentParser(description="Fit paired endpoint-label models with fixed settings.")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--release-train-labels", type=Path, required=True)
    parser.add_argument("--current-train-labels", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", choices=CLASSICAL + DEEP, default=CLASSICAL + DEEP)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    seeds = tuple(dict.fromkeys(args.seeds))
    if not seeds:
        raise SystemExit("At least one seed is required")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise SystemExit("Output must be a new or empty directory; existing runs are not resumed")
    upstream_commit = check_upstream(args.repo)
    args.output.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output / "blind_predictions.csv"
    log_path = args.output / "training_runs.jsonl"
    frozen_path = args.output / "frozen_configuration.json"

    features = pd.read_csv(args.features)
    expected_splits = {"train": 503, "calibration": 162, "test": 158}
    if features.groupby("split").size().to_dict() != expected_splits or features.file_name.duplicated().any():
        raise SystemExit("Unexpected feature cohort or duplicate file names")
    label_paths = {"release": args.release_train_labels, "current_generator": args.current_train_labels}
    labels = validate_labels(features, label_paths)
    configuration = {
        "created_utc": utc_now(),
        "protocol": "paired_label_only_controlled_refits_v13",
        "seeds": list(seeds),
        "contracts": list(CONTRACTS),
        "models": list(args.models),
        "split_counts": expected_splits,
        "ridge": {"alpha": 1000.0},
        "xgboost": XGB_PARAMETERS,
        "deep_epochs": DEEP_EPOCHS,
        "deep_optimizer": {"name": "Adam", "learning_rate": 5e-5},
        "deep_target_scaling": "common mean/std of concatenated A and B training-label vectors",
        "pairing": "same seed, initialization, row order, batch order, features, architecture, optimizer and epoch budget; training labels only differ",
        "test_labels_read": False,
        "upstream_commit": upstream_commit,
        "upstream_tracked_files_clean": True,
        "inputs": {
            "features": {"path": args.features.name, "sha256": sha256(args.features)},
            "sequences": {"path": args.sequences.name, "sha256": sha256(args.sequences)},
            "release_train_labels": {"path": "release/" + args.release_train_labels.name, "sha256": sha256(args.release_train_labels)},
            "current_train_labels": {"path": "current_generator/" + args.current_train_labels.name, "sha256": sha256(args.current_train_labels)},
            "repo": "BatteryLife",
        },
    }
    frozen_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    completed = completed_runs(log_path)

    categorical = ["dataset_group", "chemistry_system"]
    numeric = [column for column in features.columns if column not in ["split", "file_name", *categorical]]
    feature_columns = [*categorical, *numeric]
    train = features[features.split.eq("train")].copy()
    for name in [model for model in args.models if model in CLASSICAL]:
        for seed in seeds:
            for contract in CONTRACTS:
                key = (name, seed, contract)
                if key in completed:
                    continue
                started = time.perf_counter()
                try:
                    target = train.file_name.map(labels[contract]).to_numpy(float)
                    estimator = make_classical(name, seed, numeric, categorical)
                    estimator.fit(train[feature_columns], target)
                    rows = []
                    for split_name in ("calibration", "test"):
                        subset = features[features.split.eq(split_name)]
                        predicted = estimator.predict(subset[feature_columns])
                        if not np.isfinite(predicted).all():
                            raise RuntimeError("Non-finite classical predictions")
                        rows.extend({
                            "fit_contract": contract,
                            "split": split_name,
                            "dataset_group": row.dataset_group,
                            "chemistry_system": row.chemistry_system,
                            "file_name": row.file_name,
                            "model": name,
                            "seed": seed,
                            "predicted_eol": max(float(value), 1.0),
                            "prediction_kind": "train_refit_blind_fixed_control_v13",
                        } for row, value in zip(subset.itertuples(), predicted))
                    append_predictions(predictions_path, rows)
                    append_jsonl(log_path, {
                        "timestamp_utc": utc_now(),
                        "status": "completed",
                        "model": name,
                        "seed": seed,
                        "fit_contract": contract,
                        "seconds": time.perf_counter() - started,
                        "state_sha256": object_sha256(estimator),
                        "rows": len(rows),
                    })
                except Exception as exc:
                    append_jsonl(log_path, {
                        "timestamp_utc": utc_now(),
                        "status": "failed",
                        "model": name,
                        "seed": seed,
                        "fit_contract": contract,
                        "seconds": time.perf_counter() - started,
                        "error": repr(exc),
                    })
                    raise

    deep_models = [model for model in args.models if model in DEEP]
    if deep_models:
        sys.path.insert(0, str(args.repo.resolve()))
        from models import CPMLP, CPTransformer

        data = np.load(args.sequences, allow_pickle=True)
        curves = data["curves"].astype(np.float32)
        split = data["split"].astype(str)
        file_name = data["file_name"].astype(str)
        dataset_group = data["dataset_group"].astype(str)
        chemistry_system = data["chemistry_system"].astype(str)
        if curves.shape != (823, 100, 3, 300) or set(file_name) != set(features.file_name):
            raise SystemExit(f"Unexpected CyclePatch tensor or identifiers: {curves.shape}")
        train_indices = np.flatnonzero(split == "train")
        raw = {}
        for contract in CONTRACTS:
            target = np.zeros(len(curves), dtype=np.float32)
            target[train_indices] = [labels[contract][name] for name in file_name[train_indices]]
            raw[contract] = target
        pooled = np.concatenate([raw[contract][train_indices] for contract in CONTRACTS])
        common_mean = float(np.mean(pooled))
        common_std = float(np.std(pooled))
        if not np.isfinite(common_std) or common_std <= 0:
            raise SystemExit("Invalid common target scale")
        configuration["deep_common_target_mean"] = common_mean
        configuration["deep_common_target_std"] = common_std
        frozen_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable")
        device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
        for name in deep_models:
            model_config, batch_size = deep_config(name, curves.shape[1])
            model_class = CPMLP.Model if name == "cpmlp" else CPTransformer.Model
            for seed in seeds:
                for contract in CONTRACTS:
                    key = (name, seed, contract)
                    if key in completed:
                        continue
                    started = time.perf_counter()
                    try:
                        set_seed(seed)
                        model = model_class(model_config).float().to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
                        scaled = (raw[contract] - common_mean) / common_std
                        generator = torch.Generator().manual_seed(seed)
                        losses = []
                        for _ in range(DEEP_EPOCHS[name]):
                            batches = deep_loader(curves, scaled, train_indices, batch_size, True, generator)
                            losses.append(train_epoch(model, batches, optimizer, device))
                        rows = []
                        for split_name in ("calibration", "test"):
                            indices = np.flatnonzero(split == split_name)
                            predicted = deep_predict(model, curves, indices, batch_size, device) * common_std + common_mean
                            if not np.isfinite(predicted).all():
                                raise RuntimeError("Non-finite deep-model predictions")
                            rows.extend({
                                "fit_contract": contract,
                                "split": split_name,
                                "dataset_group": dataset_group[index],
                                "chemistry_system": chemistry_system[index],
                                "file_name": file_name[index],
                                "model": name,
                                "seed": seed,
                                "predicted_eol": max(float(value), 1.0),
                                "prediction_kind": "train_refit_blind_fixed_control_v13",
                            } for index, value in zip(indices, predicted))
                        append_predictions(predictions_path, rows)
                        append_jsonl(log_path, {
                            "timestamp_utc": utc_now(),
                            "status": "completed",
                            "model": name,
                            "seed": seed,
                            "fit_contract": contract,
                            "seconds": time.perf_counter() - started,
                            "device": str(device),
                            "epochs": DEEP_EPOCHS[name],
                            "epoch_losses": losses,
                            "final_loss": losses[-1],
                            "state_sha256": state_sha256(model),
                            "rows": len(rows),
                        })
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except Exception as exc:
                        append_jsonl(log_path, {
                            "timestamp_utc": utc_now(),
                            "status": "failed",
                            "model": name,
                            "seed": seed,
                            "fit_contract": contract,
                            "seconds": time.perf_counter() - started,
                            "error": repr(exc),
                        })
                        raise

    predictions = pd.read_csv(predictions_path)
    duplicate_key = ["fit_contract", "split", "file_name", "model", "seed"]
    if predictions.duplicated(duplicate_key).any():
        raise SystemExit("Duplicate prediction keys")
    expected_rows = len(CONTRACTS) * len(args.models) * len(seeds) * (162 + 158)
    if len(predictions) != expected_rows or not np.isfinite(predictions.predicted_eol).all():
        raise SystemExit(f"Incomplete controlled predictions: {len(predictions)} of {expected_rows}")
    runs = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    failures = [record for record in runs if record.get("status") == "failed"]
    manifest = {
        "timestamp_utc": utc_now(),
        "status": "PASS" if not failures else "FAIL",
        "expected_prediction_rows": expected_rows,
        "actual_prediction_rows": len(predictions),
        "completed_runs": sum(record.get("status") == "completed" for record in runs),
        "failed_runs": failures,
        "seeds": list(seeds),
        "models": list(args.models),
        "contracts": list(CONTRACTS),
        "predictions_sha256": sha256(predictions_path),
        "training_log_sha256": sha256(log_path),
        "frozen_configuration_sha256": sha256(frozen_path),
        "source_sha256": sha256(Path(__file__)),
    }
    (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
