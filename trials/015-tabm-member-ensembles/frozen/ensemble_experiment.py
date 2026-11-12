import argparse
import hashlib
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiment_analysis import load_candidates, paired_group_bootstrap


def run_ensembles(
    candidates, identity_path, output, *, core_names, families,
    previous_predictions=None, repetitions=500, four_way_groups=(),
):
    output.mkdir(parents=True, exist_ok=True)
    identity, labels, original, fingerprints = load_candidates(candidates, identity_path)
    values, hashes, aliases = {}, {}, {}
    for name, probabilities in original.items():
        normalized = probabilities.astype(np.float64)
        digest = hashlib.sha256(normalized.tobytes()).hexdigest()
        if digest in hashes:
            aliases[name] = hashes[digest]
        else:
            values[name] = normalized
            hashes[digest] = name
    names = list(values)
    core = list(dict.fromkeys(aliases.get(name, name) for name in core_names))
    if not set(core) <= set(names):
        raise ValueError("Ensemble core contains unavailable candidates")
    scores = {name: float(roc_auc_score(labels, value)) for name, value in values.items()}
    records, seen = [], set()

    def evaluate(selected, weights, kind):
        key = tuple(sorted(zip(selected, weights)))
        if key in seen:
            return
        seen.add(key)
        probabilities = sum(weight * values[name] for name, weight in key)
        records.append({
            "models": [name for name, _ in key],
            "weights": [float(weight) for _, weight in key],
            "kind": kind, "validation_auc": float(roc_auc_score(labels, probabilities)),
        })

    for name in names:
        evaluate([name], [1.0], "single")
    evaluate(names, [1 / len(names)] * len(names), "all-model-equal")
    ranked = sorted(names, key=lambda name: -scores[name])
    for count in range(2, min(len(names), 8) + 1):
        evaluate(ranked[:count], [1 / count] * count, f"top-{count}-equal")
    for family in sorted(set(families.values())):
        selected = [name for name in names if families[name] == family]
        if selected:
            evaluate(selected, [1 / len(selected)] * len(selected), f"{family}-equal")
    family_count = len(set(families[name] for name in names))
    counts = {
        family: sum(families[name] == family for name in names)
        for family in set(families.values())
    }
    evaluate(
        names, [1 / family_count / counts[families[name]] for name in names],
        "family-balanced",
    )
    for first, second in combinations(names, 2):
        for weight in [0.25, 0.5, 0.75]:
            evaluate([first, second], [weight, 1 - weight], "pair-grid")
    for selected in combinations(core, 3):
        evaluate(selected, [1 / 3] * 3, "triple-equal")
        for index in range(3):
            weights = [0.25] * 3
            weights[index] = 0.5
            evaluate(selected, weights, "triple-grid")
    for selected in four_way_groups:
        selected = [aliases.get(name, name) for name in selected]
        if len(set(selected)) != 4 or not set(selected) <= set(names):
            raise ValueError("Four-way mixtures require four available distinct candidates")
        for first, second, third in product(range(1, 6), repeat=3):
            fourth = 8 - first - second - third
            if fourth > 0:
                evaluate(
                    selected, [value / 8 for value in [first, second, third, fourth]],
                    "four-family-grid",
                )
    best = max(records, key=lambda row: row["validation_auc"])
    equal_records = [
        row for row in records
        if len(row["models"]) > 1 and np.allclose(row["weights"], row["weights"][0])
    ]
    best_equal = max(equal_records, key=lambda row: row["validation_auc"])
    best_single = ranked[0]

    def probabilities(record):
        return sum(
            weight * values[name] for name, weight in zip(record["models"], record["weights"])
        )

    best_probabilities = probabilities(best)
    for filename, record in [
        ("validation_predictions.npz", best), ("equal_weight_predictions.npz", best_equal)
    ]:
        np.savez_compressed(
            output / filename, y_true=labels, probabilities=probabilities(record),
            row_ids=identity["_row_id"].to_numpy(),
        )
    report = {
        "status": "COMPLETE", "rows": len(labels),
        "standalone_scores": scores, "duplicate_aliases": aliases,
        "unique_candidates": len(values), "configurations_evaluated": len(records),
        "four_way_groups": four_way_groups,
        "best_single": best_single, "best_equal_weight": best_equal,
        "best_ensemble": best, "candidate_artifacts": fingerprints,
        "configurations": records,
        "paired_bootstrap_vs_best_single": paired_group_bootstrap(
            labels, values[best_single], best_probabilities,
            identity["request_id"].to_numpy(), repetitions,
        ),
        "interpretation": "Post-hoc holdout-selected averaging/blending, not bootstrap retraining or independent confirmation",
        "competition_submission": False,
    }
    if previous_predictions is not None:
        previous = np.load(previous_predictions)
        np.testing.assert_array_equal(previous["y_true"], labels)
        np.testing.assert_array_equal(previous["row_ids"], identity["_row_id"].to_numpy())
        report["previous_ensemble_auc"] = float(roc_auc_score(labels, previous["probabilities"]))
        report["paired_bootstrap_vs_previous_ensemble"] = paired_group_bootstrap(
            labels, previous["probabilities"], best_probabilities,
            identity["request_id"].to_numpy(), repetitions,
        )
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in [
        "unique_candidates", "configurations_evaluated", "best_equal_weight",
        "best_ensemble", "paired_bootstrap_vs_best_single",
    ]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-predictions", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run_ensembles(
        {name: Path(path) for name, path in manifest["candidates"].items()},
        args.identities, args.output_dir, core_names=manifest["core_names"],
        families=manifest["families"], previous_predictions=args.previous_predictions,
        four_way_groups=manifest.get("four_way_groups", ()),
    )


if __name__ == "__main__":
    main()
