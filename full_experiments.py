import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from custom_network import prepare_custom_features
from feature_models import HeterogeneousClassifier
from tabm_experiment import TabMClassifier, fit_model_steps
from train import ROOT, TARGET, load_sample
from tree_experiments import run_trees


PRIMARY_CUTOFF = "2026-10-28T08:06:52.397000"
SECONDARY_CUTOFF = "2026-10-27T08:06:52.397000"
SECONDARY_END = "2026-10-27T18:00:00"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_data(path, output, *, secondary=False, cutoff_override=None, end_override=None):
    from datetime import datetime

    started = time.monotonic()
    total = pq.ParquetFile(path).metadata.num_rows
    sample = (
        load_sample(path, total, 42, expand_embeddings=True)
        .with_row_index("_row_id").sort("_time")
    )
    print(f"Loaded and sorted {sample.shape} in {time.monotonic() - started:.0f}s", flush=True)
    if sample["_time"].null_count():
        raise ValueError("Missing timestamps in experiment input")
    cutoff = datetime.fromisoformat(
        cutoff_override or (SECONDARY_CUTOFF if secondary else PRIMARY_CUTOFF)
    )
    split = int(sample["_time"].search_sorted(cutoff, side="left"))
    if secondary or end_override:
        end = int(sample["_time"].search_sorted(
            datetime.fromisoformat(end_override or SECONDARY_END), side="left"
        ))
        sample = sample.slice(0, end)
    train, valid = sample.slice(0, split), sample.slice(split)
    training_row_ids = train["_row_id"].to_numpy().copy()
    validation_row_ids = valid["_row_id"].to_numpy().copy()
    train, valid = train.drop("_row_id"), valid.drop("_row_id")
    if train.is_empty() or valid.is_empty():
        raise ValueError("The fixed temporal cutoff does not split this dataset")
    for name, frame in [("training", train), ("validation", valid)]:
        if set(frame[TARGET].unique().to_list()) != {0, 1}:
            raise ValueError(f"{name} requires both binary classes")
    x_train, c_train, x_valid, c_valid, state = prepare_custom_features(train, valid, 10)
    print(f"Prepared features {tuple(x_train.shape)} in {time.monotonic() - started:.0f}s", flush=True)
    unique_counts = train.select(pl.col(state["numeric_columns"]).n_unique()).row(0)
    continuous = [i for i, count in enumerate(unique_counts) if count > 16]
    original_train, original_valid = [], []
    category_counts = []
    original_train_array = state["encoder"].transform(
        train.select(state["categorical_columns"]).fill_null("__MISSING__").to_numpy()
    ).astype(np.int64) + 1
    original_valid_array = state["encoder"].transform(
        valid.select(state["categorical_columns"]).fill_null("__MISSING__").to_numpy()
    ).astype(np.int64) + 1
    for index, cardinality in enumerate(state["original_cardinalities"]):
        original_train.append(original_train_array[:, index].copy())
        original_valid.append(original_valid_array[:, index].copy())
        category_counts.append(np.bincount(original_train[-1], minlength=cardinality))
    # Keep identifiers outside model inputs for paired/group-aware evaluation.
    identity = (
        pl.scan_parquet(path).select("request_id", "feature1")
        .with_row_index("_row_id").collect()[validation_row_ids]
    )
    identity.write_parquet(output / "validation-identities.parquet")
    protocol = {
        "source_rows": total, "training_rows": train.height, "validation_rows": valid.height,
        "cutoff": cutoff.isoformat(), "secondary": secondary,
        "secondary_end": SECONDARY_END if secondary else None,
        "validation_end": end_override or (SECONDARY_END if secondary else None),
        "validation_labels_sha256": hashlib.sha256(
            valid[TARGET].to_numpy().astype(np.uint8).tobytes()
        ).hexdigest(),
        "validation_identities_sha256": sha256(output / "validation-identities.parquet"),
        "continuous_indices": continuous,
        "versions": {
            name: version(name)
            for name in ["torch", "tabm", "rtdl_num_embeddings", "numpy", "polars", "scikit-learn"]
        },
        "preparation_seconds": time.monotonic() - started,
        "competition_submission": False,
    }
    if not secondary and cutoff_override is None and total == 1537638:
        assert train.height == 1230110 and valid.height == 307528
    y_train = torch.tensor(train[TARGET].to_numpy().copy(), dtype=torch.float32)
    y_valid = valid[TARGET].to_numpy().copy()
    del sample, train, valid, identity
    gc.collect()
    joblib.dump(state, output / "preprocessing.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return {
        "x_train": x_train, "c_train": c_train, "x_valid": x_valid, "c_valid": c_valid,
        "y_train": y_train, "y_valid": y_valid, "state": state, "protocol": protocol,
        "original_train": original_train, "original_valid": original_valid,
        "category_counts": category_counts,
        "training_row_ids": training_row_ids,
        "validation_row_ids": validation_row_ids,
    }


def category_view(data, minimum=10, drop_high=False, *, with_mappings=False):
    state = data["state"]
    if minimum == 10 and not drop_high:
        result = (data["c_train"], data["c_valid"], state["cardinalities"])
        return (*result, state["category_remappings"]) if with_mappings else result
    train = np.empty(data["c_train"].shape, dtype=np.int64)
    valid = np.empty(data["c_valid"].shape, dtype=np.int64)
    cardinalities, mappings = [], []
    for index, counts in enumerate(data["category_counts"]):
        retained = np.flatnonzero((counts >= minimum) & (np.arange(len(counts)) != 0))
        if drop_high and state["original_cardinalities"][index] > 1000:
            retained = np.empty(0, dtype=np.int64)
        mapping = np.zeros(len(counts), dtype=np.int64)
        mapping[retained] = np.arange(1, len(retained) + 1)
        train[:, index] = mapping[data["original_train"][index]]
        valid[:, index] = mapping[data["original_valid"][index]]
        cardinalities.append(len(retained) + 1)
        mappings.append(mapping)
    result = (torch.from_numpy(train), torch.from_numpy(valid), cardinalities)
    return (*result, mappings) if with_mappings else result


def run_tabm(data, spec, output, *, device="cuda", epochs=30, interval=300, seed=42):
    name = spec["name"]
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    train_categories, valid_categories, cardinalities, mappings = category_view(
        data, spec.get("minimum", 10), spec.get("drop_high", False), with_mappings=True,
    )
    joblib.dump(mappings, destination / "category-remappings.joblib")
    dims = [
        min(32, math.ceil(cardinality ** 0.25)) if spec.get("adaptive") else spec.get("embedding_dim", 8)
        for cardinality in cardinalities
    ]
    config = {
        "n_numeric": data["x_train"].shape[1], "cardinalities": cardinalities,
        "members": spec.get("members", 8), "width": spec.get("width", 128),
        "blocks": 2, "dropout": spec.get("dropout", 0.2),
        "category_dropout": 0.3, "positive_rate": float(data["y_train"].mean().item()),
        "embedding_dimensions": dims,
        "numeric_dropout": spec.get("numeric_dropout", 0.0),
        "dropout_columns": data["protocol"]["continuous_indices"],
    }
    x_train, x_valid = data["x_train"], data["x_valid"]
    numerical_encoding = spec.get("numerical_encoding", "none")
    preprocessing_started = time.monotonic()
    bins = None
    if numerical_encoding == "ple":
        from rtdl_num_embeddings import compute_bins

        continuous = data["protocol"]["continuous_indices"]
        bins = [
            values.tolist() for values in compute_bins(
                x_train[:, continuous], n_bins=16, verbose=False
            )
        ]
    elif numerical_encoding == "quantile":
        from sklearn.preprocessing import QuantileTransformer

        continuous = data["protocol"]["continuous_indices"]
        transformer = QuantileTransformer(
            n_quantiles=min(256, len(x_train)), output_distribution="normal",
            subsample=None, random_state=42, copy=False,
        )
        train_values = transformer.fit_transform(x_train[:, continuous].numpy())
        valid_values = transformer.transform(x_valid[:, continuous].numpy())
        x_train, x_valid = x_train.clone(), x_valid.clone()
        x_train[:, continuous] = torch.from_numpy(train_values.astype(np.float32, copy=False))
        x_valid[:, continuous] = torch.from_numpy(valid_values.astype(np.float32, copy=False))
        joblib.dump(transformer, destination / "quantile-transformer.joblib")
        del train_values, valid_values
    adapted = (
        numerical_encoding in {"ple", "plr"} or spec.get("vector_projection")
        or spec.get("vector_dropout") or spec.get("vector_crosses")
        or spec.get("architecture", "tabm") != "tabm"
    )
    if adapted:
        state = data["state"]
        summary_groups, indicator_groups = [], []
        indicator_source = state["imputer"].indicator_.features_.tolist()
        for field in state["vector_fields"]:
            group = [
                i for i, column in enumerate(state["numeric_columns"])
                if column.startswith(field + "_")
            ]
            summary_groups.append(group)
            indicator_groups.append([
                len(state["numeric_columns"]) + position
                for position, source in enumerate(indicator_source) if source in group
            ])
        encoder_config = {
            "n_numeric": x_train.shape[1], "n_dense": state["n_dense"],
            "n_vectors": len(state["vector_fields"]), "cardinalities": cardinalities,
            "continuous_indices": data["protocol"]["continuous_indices"],
            "embedding_dimensions": dims,
            "numerical_encoding": numerical_encoding if numerical_encoding != "quantile" else "none",
            "bins": bins, "vector_projection": bool(spec.get("vector_projection")),
            "vector_dropout": spec.get("vector_dropout", 0.0),
            "vector_crosses": bool(spec.get("vector_crosses")),
            "summary_groups": summary_groups, "indicator_groups": indicator_groups,
            "numeric_dropout": spec.get("numeric_dropout", 0.0),
        }
        config = {
            "encoder_config": encoder_config, "architecture": spec.get("architecture", "tabm"),
            "width": spec.get("width", 128), "dropout": spec.get("dropout", 0.2),
            "members": spec.get("members", 8),
            "positive_rate": float(data["y_train"].mean().item()),
            "rank": spec.get("rank", 32), "experts": spec.get("experts", 1),
            "gating": spec.get("gating", True),
        }
        model = HeterogeneousClassifier(**config).to(device)
    else:
        model = TabMClassifier(**config).to(device)
    learning_rate = spec.get("learning_rate", 0.0005)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=spec.get("weight_decay", 0.01)
    )
    scheduler = None
    if spec.get("cosine"):
        period = 2 * math.ceil(len(data["y_train"]) / 1024)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.1 + 0.45 * (1 + math.cos(math.pi * min(step / period, 1)))
        )
    variant_preparation_seconds = time.monotonic() - preprocessing_started
    print(f"START {name}: {json.dumps(spec)}", flush=True)
    metrics = fit_model_steps(
        model, config, optimizer, x_train, train_categories, data["y_train"],
        x_valid, valid_categories, data["y_valid"],
        device=device, output=destination, epochs=epochs, patience=5, batch_size=1024,
        validation_interval=interval, seed=seed, scheduler=scheduler,
        fixed_budget=spec.get("fixed_budget", False),
    )
    saved_predictions = np.load(destination / "validation_predictions.npz")
    np.savez_compressed(
        destination / "validation_predictions.npz",
        y_true=saved_predictions["y_true"],
        probabilities=saved_predictions["probabilities"],
        row_ids=data["validation_row_ids"],
    )
    result = {
        "name": name, "spec": spec, "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": config, **metrics,
        "variant_preparation_seconds": variant_preparation_seconds,
    }
    (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT {name}: AUC={result['validation_auc']:.10f}", flush=True)
    del model, optimizer, train_categories, valid_categories, x_train, x_valid
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def optimization_specs():
    return [
        {"name": "epoch-checkpoint-control", "epoch_only": True},
        {"name": "reference"},
        {"name": "learning-rate-1e4", "learning_rate": 0.0001},
        {"name": "dropout-035", "dropout": 0.35},
        {"name": "weight-decay-003", "weight_decay": 0.03},
        {"name": "width-64", "width": 64},
        {"name": "numeric-dropout-01", "numeric_dropout": 0.1},
        {"name": "cosine-two-epochs", "cosine": True},
        {"name": "minimum-category-50", "minimum": 50},
        {"name": "embedding-dim-4", "embedding_dim": 4},
        {"name": "adaptive-category-dims", "adaptive": True},
        {"name": "without-high-cardinality", "drop_high": True},
    ]


def feature_specs():
    return [
        {"name": "reference"},
        {"name": "ple-16bins-d8", "numerical_encoding": "ple"},
        {"name": "plr-d8", "numerical_encoding": "plr"},
        {"name": "quantile-normal", "numerical_encoding": "quantile"},
        {"name": "vector-projection-ln", "vector_projection": True},
        {"name": "vector-dropout-01", "vector_dropout": 0.1},
        {"name": "vector-crosses", "vector_crosses": True},
    ]


def architecture_specs(base):
    return [
        {**base, "name": "ensemble-16", "members": 16},
        {**base, "name": "ensemble-32", "members": 32},
        {**base, "name": "dcnv2-r32", "architecture": "dcnv2", "rank": 32},
        {**base, "name": "dcnv2-r64", "architecture": "dcnv2", "rank": 64},
        {**base, "name": "dcnv2-moe4", "architecture": "dcnv2", "rank": 32, "experts": 4},
        {**base, "name": "finalmlp-gated", "architecture": "finalmlp", "gating": True},
        {**base, "name": "finalmlp-ungated", "architecture": "finalmlp", "gating": False},
        {
            **base, "name": "finalmlp-alternate-lr", "architecture": "finalmlp",
            "gating": True, "learning_rate": 0.0001 if base.get("learning_rate", 0.0005) != 0.0001 else 0.0005,
        },
    ]


def run_confirmation(path, output, specs, *, device="cuda", epochs=30, interval=300, fold=None):
    report = {"stage": "confirmation", "folds": [], "results": [], "competition_submission": False}
    if fold is None:
        spec_path = output / "confirmation-specs.json"
        spec_path.write_text(json.dumps(specs, indent=2) + "\n")
        worker = globals().get("FROZEN_EXPERIMENT_FILE", Path(__file__).resolve())
        report["summary"] = []
        for selected in ["primary", "earlier"]:
            subprocess.run([
                sys.executable, str(worker), "--output-dir", str(output),
                "--device", device, "--stage", "confirmation",
                "--spec-file", str(spec_path), "--confirmation-fold", selected,
                "--confirmation-input-file", str(path),
                "--epochs", str(epochs), "--validation-interval", str(interval),
            ], check=True)
            partial = json.loads((output / f"{selected}-results.json").read_text())
            if partial.get("status") != "COMPLETE":
                raise ValueError(f"Incomplete confirmation fold: {selected}")
            for key in ["folds", "results", "summary"]:
                report[key].extend(partial[key])
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "COMPLETE"
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    if fold not in {"primary", "earlier"}:
        raise ValueError(f"Unknown confirmation fold: {fold}")
    report_path = output / f"{fold}-results.json"
    for secondary in [fold == "earlier"]:
        fold = "earlier" if secondary else "primary"
        destination = output / fold
        destination.mkdir(parents=True, exist_ok=True)
        data = prepare_data(path, destination, secondary=secondary)
        report["folds"].append({"name": fold, **data["protocol"]})
        for recipe in specs:
            for seed in [42, 43, 44]:
                spec = {**recipe, "name": f"{recipe['name']}-seed-{seed}"}
                result = run_tabm(
                    data, spec, destination, device=device, epochs=epochs, interval=interval, seed=seed,
                )
                result["fold"] = fold
                result["recipe"] = recipe["name"]
                report["results"].append(result)
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        del data
        gc.collect()
    report["summary"] = []
    for fold in [entry["name"] for entry in report["folds"]]:
        for recipe in specs:
            values = [
                row["validation_auc"] for row in report["results"]
                if row["fold"] == fold and row["recipe"] == recipe["name"]
            ]
            report["summary"].append({
                "fold": fold, "recipe": recipe["name"], "mean_auc": float(np.mean(values)),
                "std_auc": float(np.std(values, ddof=1)), "seed_aucs": values,
            })
    report["status"] = "COMPLETE"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def fit_stacking_base(path, output, plan, cutoff, end, base, *, device="cuda"):
    output.mkdir(parents=True, exist_ok=True)
    if base == "neural":
        data = prepare_data(path, output, cutoff_override=cutoff, end_override=end)
        recipe = {
            **plan["neural_recipe"], "name": "neural-fixed-budget", "fixed_budget": True,
        }
        return run_tabm(
            data, recipe, output, device=device,
            epochs=max(1, int(math.ceil(plan["fixed_neural_epochs"]))),
            interval=300, seed=42,
        )
    if base != "tree":
        raise ValueError(f"Unknown stacking base: {base}")
    tree_output = output / "tree"
    tree_output.mkdir(exist_ok=True)
    if plan["tree_kind"] == "lightgbm":
        from local_tree_experiment import run as run_local_tree

        identity = output / "validation-identities.parquet"
        return run_local_tree(
            path, tree_output, iterations=plan["tree_iterations"],
            reference_identity=identity if identity.is_file() else None,
            cutoff_string=cutoff, validation_end=end, fixed_budget=True,
        )
    return run_trees(
        path, tree_output, cutoff, device=device,
        iterations=plan["tree_iterations"], light_iterations=plan["tree_iterations"],
        only=[plan["tree_kind"]], valid_end=end, fixed_budget=True,
    )


def finish_stacking(output, report):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score

    features, labels, identities, outer = [], [], [], None
    expected_periods = [
        ("2026-10-26T12:00:00", "2026-10-27T00:00:00"),
        ("2026-10-27T00:00:00", "2026-10-27T12:00:00"),
        ("2026-10-27T12:00:00", PRIMARY_CUTOFF),
        (PRIMARY_CUTOFF, None),
    ]
    if [(fold["cutoff"], fold["end"]) for fold in report["folds"]] != expected_periods:
        raise ValueError("Stacking requires all three forward OOF blocks and the outer holdout")
    for index, fold in enumerate(report["folds"]):
        destination = output / ("outer" if fold["end"] is None else f"oof-{index + 1}")
        with (
            np.load(destination / fold["neural"]["name"] / "validation_predictions.npz") as neural,
            np.load(destination / "tree" / fold["tree"]["name"] / "validation_predictions.npz") as tree,
        ):
            np.testing.assert_array_equal(neural["y_true"], tree["y_true"])
            np.testing.assert_array_equal(neural["row_ids"], tree["row_ids"])
            target, row_ids = neural["y_true"].copy(), neural["row_ids"].copy()
            probability_matrix = np.column_stack([
                neural["probabilities"], tree["probabilities"],
            ])
        clipped = np.clip(probability_matrix, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        if fold["end"] is None:
            outer = (logits, target, row_ids, probability_matrix)
        else:
            features.append(logits)
            labels.append(target)
            identities.append(row_ids)
    combined_ids = np.concatenate(identities)
    if len(np.unique(combined_ids)) != len(combined_ids):
        raise ValueError("OOF prediction blocks overlap")
    if len(np.unique(outer[2])) != len(outer[2]) or np.intersect1d(combined_ids, outer[2]).size:
        raise ValueError("Outer holdout identities duplicate or overlap OOF rows")
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(np.concatenate(features), np.concatenate(labels))
    joblib.dump(meta, output / "meta-model.joblib")
    probabilities = meta.predict_proba(outer[0])[:, 1]
    np.savez_compressed(
        output / "validation_predictions.npz", y_true=outer[1],
        probabilities=probabilities, row_ids=outer[2],
    )
    np.savez_compressed(
        output / "oof_predictions.npz", logits=np.concatenate(features),
        y_true=np.concatenate(labels), row_ids=combined_ids,
    )
    report.update({
        "stage": "stacking", "status": "COMPLETE", "oof_rows": len(combined_ids),
        "validation_rows": len(outer[1]),
        "validation_auc": float(roc_auc_score(outer[1], probabilities)),
        "validation_logloss": float(log_loss(outer[1], probabilities, labels=[0, 1])),
        "matched_neural_auc": float(roc_auc_score(outer[1], outer[3][:, 0])),
        "matched_tree_auc": float(roc_auc_score(outer[1], outer[3][:, 1])),
        "matched_equal_blend_auc": float(roc_auc_score(outer[1], outer[3].mean(axis=1))),
        "meta_coefficients": meta.coef_.tolist(), "meta_intercept": meta.intercept_.tolist(),
    })
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def run_stacking(path, output, plan, *, device="cuda", outer_only=False):
    periods = [
        ("2026-10-26T12:00:00", "2026-10-27T00:00:00"),
        ("2026-10-27T00:00:00", "2026-10-27T12:00:00"),
        ("2026-10-27T12:00:00", PRIMARY_CUTOFF),
        (PRIMARY_CUTOFF, None),
    ]
    fixed_epochs = max(1, int(math.ceil(plan["fixed_neural_epochs"])))
    report = {
        "stage": "stacking-base-fits" if outer_only else "stacking",
        "folds": [], "competition_submission": False, "fixed_neural_epochs": fixed_epochs,
    }
    plan_path = output / "stacking-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    worker = globals().get("FROZEN_EXPERIMENT_FILE", Path(__file__).resolve())
    for index, (cutoff, end) in enumerate(periods):
        if outer_only and end is not None:
            continue
        destination = output / ("outer" if end is None else f"oof-{index + 1}")
        destination.mkdir(parents=True, exist_ok=True)
        for base in ["neural", "tree"]:
            subprocess.run([
                sys.executable, str(worker), "--data-dir", str(path.parent),
                "--output-dir", str(destination), "--device", device,
                "--stage", "stacking", "--spec-file", str(plan_path),
                "--stacking-base", base, "--stacking-input-file", str(path),
                "--cutoff", cutoff,
                *(["--validation-end", end] if end else []),
            ], check=True)
        protocol = json.loads((destination / "protocol.json").read_text())
        neural = json.loads((destination / "neural-fixed-budget/results.json").read_text())
        tree_report = json.loads((destination / "tree/results.json").read_text())
        report["folds"].append({
            "cutoff": cutoff, "end": end, "protocol": protocol,
            "neural": neural, "tree": tree_report["results"][0],
        })
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if outer_only:
        report["status"] = "COMPLETE"
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    return finish_stacking(output, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--stage", choices=["optimization", "features", "trees", "recipes", "combined", "confirmation", "stacking"],
        default="optimization",
    )
    parser.add_argument("--spec-file", type=Path)
    parser.add_argument("--secondary", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", action="append")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-interval", type=int, default=300)
    parser.add_argument("--stacking-base", choices=["neural", "tree"])
    parser.add_argument("--stacking-input-file", type=Path)
    parser.add_argument("--confirmation-fold", choices=["primary", "earlier"])
    parser.add_argument("--confirmation-input-file", type=Path)
    parser.add_argument("--cutoff")
    parser.add_argument("--validation-end")
    parser.add_argument("--outer-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The full Kaggle experiment requires CUDA")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "trees":
        run_trees(
            args.data_dir / "train.parquet", args.output_dir, PRIMARY_CUTOFF,
            device=args.device, only=args.only,
            worker_path=globals().get("FROZEN_LIGHTGBM_WORKER_FILE"),
        )
        return
    if args.stage in {"confirmation", "stacking"}:
        if args.spec_file is None:
            parser.error("--spec-file is required for confirmation/stacking")
        plan = json.loads(args.spec_file.read_text())
        if args.stage == "confirmation":
            run_confirmation(
                args.confirmation_input_file or args.data_dir / "train.parquet",
                args.output_dir, plan,
                device=args.device, epochs=args.epochs, interval=args.validation_interval,
                fold=args.confirmation_fold,
            )
        else:
            if args.stacking_base:
                if args.cutoff is None:
                    parser.error("--cutoff is required with --stacking-base")
                fit_stacking_base(
                    args.stacking_input_file or args.data_dir / "train.parquet",
                    args.output_dir, plan,
                    args.cutoff, args.validation_end, args.stacking_base, device=args.device,
                )
            else:
                run_stacking(
                    args.data_dir / "train.parquet", args.output_dir, plan,
                    device=args.device, outer_only=args.outer_only,
                )
        return
    data = prepare_data(
        args.data_dir / "train.parquet", args.output_dir, secondary=args.secondary
    )
    report = {"stage": args.stage, "protocol": data["protocol"], "results": [], "competition_submission": False}
    if args.stage in {"recipes", "combined"}:
        if args.spec_file is None:
            parser.error("--spec-file is required for recipes")
        specs = json.loads(args.spec_file.read_text())
    else:
        specs = optimization_specs() if args.stage == "optimization" else feature_specs()
    if args.only:
        selected = set(args.only)
        names = {spec["name"] for spec in specs}
        if not selected <= names:
            raise ValueError(f"Unknown experiment names: {selected - names}")
        specs = [spec for spec in specs if spec["name"] in selected]
    for spec in specs:
        result = run_tabm(
            data, spec, args.output_dir, device=args.device,
            epochs=args.epochs,
            interval=math.ceil(len(data["y_train"]) / 1024) if spec.get("epoch_only") else args.validation_interval,
            seed=spec.get("seed", args.seed),
        )
        report["results"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.stage == "combined":
        best = max(report["results"], key=lambda entry: entry["validation_auc"])
        report["selected_combination"] = {"name": best["name"], "spec": best["spec"]}
        for spec in architecture_specs(best["spec"]):
            result = run_tabm(
                data, spec, args.output_dir, device=args.device,
                epochs=args.epochs, interval=args.validation_interval,
            )
            report["results"].append(result)
            (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "COMPLETE"
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
