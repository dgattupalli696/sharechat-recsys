import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def auc_order(labels, probabilities):
    order = np.argsort(probabilities, kind="stable")
    values = probabilities[order]
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    return order, starts, labels[order].astype(np.float64)


def weighted_auc(prepared, weights):
    order, starts, positive = prepared
    weighted = weights[order]
    positives = np.add.reduceat(weighted * positive, starts)
    negatives = np.add.reduceat(weighted * (1 - positive), starts)
    denominator = positives.sum() * negatives.sum()
    if denominator == 0:
        raise ValueError("AUC bootstrap replicate contains only one class")
    return float(np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives)) / denominator)


def paired_group_bootstrap(labels, reference, candidate, groups, repetitions=500):
    _, inverse = np.unique(groups, return_inverse=True)
    group_count = int(inverse.max()) + 1
    probabilities = np.full(group_count, 1 / group_count)
    reference_order = auc_order(labels, reference)
    candidate_order = auc_order(labels, candidate)
    rng = np.random.default_rng(2026)
    deltas, excluded = [], 0
    for _ in range(repetitions * 10):
        counts = rng.multinomial(group_count, probabilities)
        weights = counts[inverse].astype(np.float64)
        if not (weights[labels == 0].sum() and weights[labels == 1].sum()):
            excluded += 1
            continue
        deltas.append(
            weighted_auc(candidate_order, weights) - weighted_auc(reference_order, weights)
        )
        if len(deltas) == repetitions:
            break
    if len(deltas) != repetitions:
        raise ValueError("Could not obtain enough two-class group-bootstrap replicates")
    return {
        "group_count": group_count, "repetitions": repetitions,
        "excluded_single_class_draws": excluded,
        "mean_delta": float(np.mean(deltas)),
        "percentile_95_interval": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "interpretation": "request-clustered, conditional on selected models/weights; not selection-adjusted or time-block adjusted",
    }


def load_candidates(candidates, identity_path):
    identity = pl.read_parquet(identity_path)
    row_ids = identity["_row_id"].to_numpy()
    if len(np.unique(row_ids)) != len(row_ids) or identity["request_id"].null_count():
        raise ValueError("Evaluation identities require unique row IDs and nonmissing request IDs")
    reference_y, values, fingerprints = None, {}, {}
    for name, path in candidates.items():
        arrays = np.load(path)
        y, probabilities = arrays["y_true"], arrays["probabilities"]
        if reference_y is None:
            reference_y = y.copy()
        np.testing.assert_array_equal(y, reference_y)
        if "row_ids" in arrays:
            np.testing.assert_array_equal(arrays["row_ids"], row_ids)
        else:
            batch_identity = path.parent.parent / "validation-identities.parquet"
            if not batch_identity.is_file():
                raise ValueError(f"{name} has no row IDs or batch identity artifact")
            np.testing.assert_array_equal(
                pl.read_parquet(batch_identity)["_row_id"].to_numpy(), row_ids
            )
        if len(probabilities) != len(identity) or not np.isfinite(probabilities).all():
            raise ValueError(f"{name} has invalid or unaligned predictions")
        if not ((probabilities >= 0).all() and (probabilities <= 1).all()):
            raise ValueError(f"{name} predictions are not probabilities")
        values[name] = probabilities.copy()
        with path.open("rb") as stream:
            fingerprints[name] = {
                "path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()
            }
    if len(values) < 2:
        raise ValueError("Blending requires at least two candidates")
    return identity, reference_y, values, fingerprints


def analyze(candidates, identity_path, output, *, repetitions=500):
    output.mkdir(parents=True, exist_ok=True)
    identity, reference_y, values, fingerprints = load_candidates(candidates, identity_path)
    row_ids = identity["_row_id"].to_numpy()
    scores = {
        name: float(roc_auc_score(reference_y, probabilities))
        for name, probabilities in values.items()
    }
    best_single = max(scores, key=scores.get)
    best_score, best_probabilities = scores[best_single], values[best_single]
    best = {"models": [best_single], "weights": [1.0], "validation_auc": best_score}
    grid = []
    names = list(values)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            for weight in [0.25, 0.5, 0.75]:
                probabilities = weight * values[first] + (1 - weight) * values[second]
                auc = float(roc_auc_score(reference_y, probabilities))
                record = {
                    "models": [first, second], "weights": [weight, 1 - weight],
                    "validation_auc": auc,
                }
                grid.append(record)
                if auc > best_score:
                    best_score, best, best_probabilities = auc, record, probabilities
    np.savez_compressed(
        output / "validation_predictions.npz",
        y_true=reference_y, probabilities=best_probabilities, row_ids=row_ids,
    )
    report = {
        "rows": len(reference_y),
        "standalone_scores": scores, "best_single": best_single,
        "best_blend": best, "grid": grid, "candidate_artifacts": fingerprints,
        "paired_bootstrap_vs_best_single": paired_group_bootstrap(
            reference_y, values[best_single], best_probabilities,
            identity["request_id"].to_numpy(), repetitions,
        ),
        "competition_submission": False,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidates = {
        name: Path(path) for name, path in json.loads(args.manifest.read_text()).items()
    }
    report = analyze(candidates, args.identities, args.output_dir)
    print(json.dumps({
        "best_single": report["best_single"],
        "best_blend": report["best_blend"],
        "bootstrap": report["paired_bootstrap_vs_best_single"],
    }, indent=2))


if __name__ == "__main__":
    main()
