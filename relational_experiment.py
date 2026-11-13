import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from feature_models import HeterogeneousClassifier
from full_experiments import category_view, prepare_data
from relational_audit import EARLIER, EARLIER_END, PRIMARY, history_features
from tabm_experiment import fit_model_steps


METHODS = ["tabm-control", "history-tabm", "matrix-factorization", "node-only-tabm", "graphsage-tabm"]


def graph_arrays(left, right, cardinalities, values, fanout=8, seed=42):
    if fanout <= 0 or len(left) != len(right) or len(left) != len(values):
        raise ValueError("Invalid graph dimensions")
    n_left, n_right = cardinalities
    if min(n_left, n_right) < 2:
        raise ValueError("Both graph partitions require known nodes")
    if np.any(left < 0) or np.any(left >= n_left) or np.any(right < 0) or np.any(right >= n_right):
        raise ValueError("Entity index exceeds training vocabulary")
    right_global = np.where(right > 0, right + n_left, 0)
    complete = (left > 0) & (right > 0)
    first, second = left[complete], right_global[complete]
    size = n_left + n_right
    adjacency = csr_matrix(
        (np.ones(2 * len(first), dtype=np.float32),
         (np.concatenate([first, second]), np.concatenate([second, first]))),
        shape=(size, size),
    )
    adjacency.sum_duplicates()
    degree = np.diff(adjacency.indptr)
    neighbors = np.zeros((size, fanout), dtype=np.int64)
    known = np.flatnonzero(degree)
    rng = np.random.default_rng(seed)
    for column in range(fanout):
        offsets = (rng.random(len(known)) * degree[known]).astype(np.int64)
        neighbors[known, column] = adjacency.indices[adjacency.indptr[known] + offsets]
    attributes = np.zeros((size, values.shape[1] + 3), dtype=np.float32)
    for ids in [left, right_global]:
        present = ids > 0
        counts = np.bincount(ids[present], minlength=size)
        nodes = np.flatnonzero(counts)
        for column in range(values.shape[1]):
            sums = np.bincount(ids[present], weights=values[present, column], minlength=size)
            attributes[nodes, column] = sums[nodes] / counts[nodes]
    attributes[:, -3] = np.log1p(degree)
    attributes[1:n_left, -2] = 1
    attributes[n_left + 1:, -1] = 1
    return {
        "attributes": attributes, "neighbors": neighbors,
        "degree": degree.astype(np.int64), "n_left": n_left,
        "training_unique_edges": int(adjacency.nnz // 2),
    }


class MatrixFactorization(nn.Module):
    def __init__(self, cardinalities, positive_rate, dimension=32):
        super().__init__()
        self.factors = nn.ModuleList(nn.Embedding(n, dimension, padding_idx=0) for n in cardinalities)
        self.biases = nn.ModuleList(nn.Embedding(n, 1, padding_idx=0) for n in cardinalities)
        for layer in self.factors:
            nn.init.normal_(layer.weight, std=.02)
            with torch.no_grad():
                layer.weight[0].zero_()
        for layer in self.biases:
            nn.init.zeros_(layer.weight)
        self.intercept = nn.Parameter(torch.tensor(float(np.log(positive_rate / (1 - positive_rate)))))

    def forward(self, numeric, categorical):
        left, right = categorical[:, -2], categorical[:, -1]
        score = (self.factors[0](left) * self.factors[1](right)).sum(1, keepdim=True)
        return score + self.biases[0](left) + self.biases[1](right) + self.intercept


class GraphTabM(HeterogeneousClassifier):
    def __init__(self, base_config, graph, attribute_columns, *, use_neighbors):
        super().__init__(**base_config)
        from tabm import TabM

        self.use_neighbors = use_neighbors
        self.n_left = int(graph["n_left"])
        self.native_categories = len(base_config["encoder_config"]["cardinalities"])
        self.register_buffer("attributes", torch.from_numpy(graph["attributes"]))
        self.register_buffer("neighbors", torch.from_numpy(graph["neighbors"]))
        self.register_buffer("attribute_columns", torch.tensor(attribute_columns, dtype=torch.int64))
        self.entity_embedding = nn.Embedding(len(self.attributes), 16, padding_idx=0)
        nn.init.normal_(self.entity_embedding.weight, std=.02)
        with torch.no_grad():
            self.entity_embedding.weight[0].zero_()
        dimension = self.attributes.shape[1] + 16
        self.self_projection = nn.Linear(dimension, 32)
        self.neighbor_projection = nn.Linear(dimension, 32, bias=False)
        self.backbone = TabM.make(
            n_num_features=self.encoder.output_dim + 128, d_out=1,
            k=base_config["members"], n_blocks=2,
            d_block=base_config["width"], dropout=base_config["dropout"],
        )
        with torch.no_grad():
            rate = base_config["positive_rate"]
            self.backbone.output.bias.fill_(float(np.log(rate / (1 - rate))))

    def node_input(self, ids):
        return torch.cat([self.attributes[ids], self.entity_embedding(ids)], dim=-1)

    def encode_node(self, ids, numeric, side):
        attributes = self.attributes[ids]
        fallback = torch.cat([
            numeric[:, self.attribute_columns],
            numeric.new_zeros((len(numeric), 1)),
            numeric.new_full((len(numeric), 1), float(side == 0)),
            numeric.new_full((len(numeric), 1), float(side == 1)),
        ], dim=1)
        attributes = torch.where((ids == 0).unsqueeze(1), fallback, attributes)
        own = torch.cat([attributes, self.entity_embedding(ids)], dim=1)
        if self.use_neighbors:
            neighbors = self.neighbors[ids]
            mask = (neighbors > 0).unsqueeze(-1)
            pooled = (self.node_input(neighbors) * mask).sum(1) / mask.sum(1).clamp_min(1)
        else:
            pooled = torch.zeros_like(own)
        return F.normalize(F.relu(self.self_projection(own) + self.neighbor_projection(pooled)), dim=1)

    def forward(self, numeric, categorical):
        left = categorical[:, -2]
        raw_right = categorical[:, -1]
        right = torch.where(raw_right > 0, raw_right + self.n_left, 0)
        first = self.encode_node(left, numeric, 0)
        second = self.encode_node(right, numeric, 1)
        graph_features = torch.cat([first, second, first * second, (first - second).abs()], dim=1)
        encoded = self.encoder(numeric, categorical[:, :self.native_categories])
        return self.backbone(torch.cat([encoded, graph_features], dim=1)).squeeze(-1)


def base_configuration(data, cardinalities, history_width=0):
    state = data["state"]
    groups, indicators = [], []
    sources = state["imputer"].indicator_.features_.tolist()
    for field in state["vector_fields"]:
        group = [i for i, name in enumerate(state["numeric_columns"]) if name.startswith(field + "_")]
        groups.append([i + history_width for i in group])
        indicators.append([
            len(state["numeric_columns"]) + position + history_width
            for position, source in enumerate(sources) if source in group
        ])
    continuous = list(range(history_width)) + [
        i + history_width for i in data["protocol"]["continuous_indices"]
    ]
    return {
        "encoder_config": {
            "n_numeric": data["x_train"].shape[1] + history_width,
            "n_dense": state["n_dense"] + history_width,
            "n_vectors": len(state["vector_fields"]), "cardinalities": cardinalities,
            "continuous_indices": continuous, "embedding_dimensions": [8] * len(cardinalities),
            "numerical_encoding": "plr", "vector_crosses": True,
            "summary_groups": groups, "indicator_groups": indicators,
        },
        "architecture": "tabm", "width": 128, "dropout": .2, "members": 8,
        "positive_rate": float(data["y_train"].mean().item()),
    }


def run_fold(path, output, fields, fold, epochs=8, device="cuda"):
    started = time.monotonic()
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_data(path, output, secondary=fold == "earlier")
    raw = pl.read_parquet(path, columns=["feature1", "watched", *fields])
    train = raw[data["training_row_ids"]]
    valid = raw[data["validation_row_ids"]]
    np.testing.assert_array_equal(valid["watched"].to_numpy(), data["y_valid"])
    cutoff = EARLIER if fold == "earlier" else PRIMARY
    assert train["feature1"].max() < cutoff <= valid["feature1"].min()
    if fold == "earlier":
        assert valid["feature1"].max() < EARLIER_END
    train_entities, valid_entities, cardinalities = [], [], []
    for field in fields:
        index = data["state"]["categorical_columns"].index(field)
        a = data["original_train"][index].copy()
        b = data["original_valid"][index].copy()
        a[train[field].is_null().to_numpy()] = 0
        b[valid[field].is_null().to_numpy()] = 0
        train_entities.append(a)
        valid_entities.append(b)
        cardinalities.append(data["state"]["original_cardinalities"][index])
    entity_train = torch.from_numpy(np.stack(train_entities, axis=1))
    entity_valid = torch.from_numpy(np.stack(valid_entities, axis=1))
    np.savez_compressed(
        output / "validation-entity-coverage.npz",
        row_ids=data["validation_row_ids"],
        left_seen=valid_entities[0] > 0, right_seen=valid_entities[1] > 0,
    )
    attribute_columns = data["protocol"]["continuous_indices"][:16]
    graph = graph_arrays(
        *train_entities, cardinalities,
        data["x_train"][:, attribute_columns].numpy(),
    )
    np.savez_compressed(output / "graph-data.npz", **graph, attribute_columns=attribute_columns)
    history_train, history_valid = history_features(train, valid, fields)
    scaler = StandardScaler()
    history_train = scaler.fit_transform(history_train).astype(np.float32)
    history_valid = scaler.transform(history_valid).astype(np.float32)
    np.clip(history_train, -10, 10, out=history_train)
    np.clip(history_valid, -10, 10, out=history_valid)
    import joblib

    joblib.dump(scaler, output / "history-scaler.joblib")
    masked_train, masked_valid, masked_cards, mappings = category_view(
        data, drop_high=True, with_mappings=True,
    )
    joblib.dump(mappings, output / "category-remappings.joblib")
    categories_train = torch.cat([masked_train, entity_train], dim=1)
    categories_valid = torch.cat([masked_valid, entity_valid], dim=1)
    protocol = {
        **data["protocol"], "fold": fold, "entity_fields": fields,
        "entity_semantics": "anonymous entity-like categories, not confirmed users/items",
        "entity_cardinalities": cardinalities,
        "validation_seen_fraction": {
            name: float((values > 0).mean()) for name, values in zip(fields, valid_entities)
        },
        "validation_both_seen_fraction": float(((valid_entities[0] > 0) & (valid_entities[1] > 0)).mean()),
        "graph_edges": graph["training_unique_edges"], "graph_uses_validation_rows": False,
        "graph_uses_labels": False, "neighbor_fanout": 8, "graphsage_layers": 1,
        "node_attributes": "training means of first 16 standardized continuous covariates, log degree and partition type",
        "unseen_node_attributes": "current row covariates; no held-out neighbors or label updates",
        "history": "strictly earlier timestamps for training; frozen training-prefix aggregates for validation; prior .05, strength 20",
        "epochs": epochs, "seed": 42, "model_selection": "fixed-budget endpoint",
        "test_data_used": False, "competition_submission": False,
        "relational_preparation_seconds": time.monotonic() - started,
    }
    (output / "relational-protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    del raw, train, valid
    gc.collect()
    results = []
    for method in METHODS:
        destination = output / method
        destination.mkdir()
        torch.manual_seed(42)
        config = base_configuration(data, masked_cards)
        x_train, x_valid = data["x_train"], data["x_valid"]
        c_train, c_valid = masked_train, masked_valid
        if method == "history-tabm":
            width = history_train.shape[1]
            config = base_configuration(data, masked_cards, width)
            x_train = torch.cat([torch.from_numpy(history_train), x_train], dim=1)
            x_valid = torch.cat([torch.from_numpy(history_valid), x_valid], dim=1)
            model = HeterogeneousClassifier(**config)
        elif method == "matrix-factorization":
            config = {"cardinalities": cardinalities, "positive_rate": config["positive_rate"], "dimension": 32}
            model = MatrixFactorization(**config)
            c_train, c_valid = entity_train, entity_valid
            x_train = torch.zeros((len(entity_train), 1))
            x_valid = torch.zeros((len(entity_valid), 1))
        elif method in {"node-only-tabm", "graphsage-tabm"}:
            model = GraphTabM(config, graph, attribute_columns, use_neighbors=method == "graphsage-tabm")
            c_train, c_valid = categories_train, categories_valid
        else:
            model = HeterogeneousClassifier(**config)
        model = model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=.001 if method == "matrix-factorization" else .0005,
            weight_decay=.01,
        )
        print(f"START {fold}/{method}: all {len(x_train):,} training rows, {epochs} epochs", flush=True)
        metrics = fit_model_steps(
            model, config, optimizer, x_train, c_train, data["y_train"],
            x_valid, c_valid, data["y_valid"], device=device, output=destination,
            epochs=epochs, patience=epochs, batch_size=1024, validation_interval=300,
            seed=42, fixed_budget=True,
        )
        with np.load(destination / "validation_predictions.npz") as arrays:
            predictions = arrays["probabilities"].copy()
        np.savez_compressed(
            destination / "validation_predictions.npz", y_true=data["y_valid"],
            row_ids=data["validation_row_ids"], probabilities=predictions,
        )
        result = {
            "method": method, "fold": fold, "model_config": config,
            "parameters": sum(p.numel() for p in model.parameters()),
            "train_rows": len(x_train), "validation_rows": len(x_valid), **metrics,
        }
        (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"RESULT {fold}/{method}: AUC {metrics['validation_auc']:.10f}", flush=True)
        del model, optimizer, x_train, x_valid, c_train, c_valid
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields", nargs=2, required=True)
    parser.add_argument("--fold", choices=["primary", "earlier"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epoch count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA device is unavailable")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.fold:
        run_fold(args.input, args.output / args.fold, args.fields, args.fold, args.epochs, args.device)
    else:
        for fold in ["primary", "earlier"]:
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "--input", str(args.input),
                "--output", str(args.output), "--fields", *args.fields,
                "--fold", fold, "--epochs", str(args.epochs), "--device", args.device,
            ], check=True)
        summary = {
            "status": "COMPLETE", "competition_submission": False,
            "folds": {
                fold: json.loads((args.output / fold / "results.json").read_text())
                for fold in ["primary", "earlier"]
            },
        }
        (args.output / "relational-results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
