"""T050 data preparation for `multitower.py` (no torch import; see that module).

Writes quantile-binned T036 matrices, raw-ID codes (frequent fit-row values only) and XGBoost
depth12 leaf indices from a fixed-round fit on pre-cutoff rows only.
"""
import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np

ID_COLUMNS = ["feature5", "feature18", "feature23", "feature20"]
BINS = 255


def quantile_grid(x, bins=BINS, sample=200_000, seed=0):
    """Per-column quantile edges (columns, bins) from a row sample; all-missing columns get zeros."""
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(len(x), min(sample, len(x)), replace=False))
    values = np.asarray(x[rows], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        grid = np.nanquantile(values, np.linspace(0, 1, bins), axis=0).T
    return np.ascontiguousarray(np.nan_to_num(grid, nan=0.0), dtype=np.float32)


def bin_matrix(x, grid, chunk=100_000):
    """int16 bin index in [0, bins] per value (count of edges <= value); missing -> -1."""
    out = np.empty(x.shape, dtype=np.int16)
    for start in range(0, len(x), chunk):
        block = np.asarray(x[start:start + chunk], dtype=np.float32)
        for c in range(block.shape[1]):
            column = block[:, c]
            codes = np.searchsorted(grid[c], column, side="right").astype(np.int16)
            codes[np.isnan(column)] = -1
            out[start:start + chunk, c] = codes
    return out


def id_codes(fit, valid, minimum=3):
    """Codes 1..k for values seen >= minimum times in fit rows; everything else 0."""
    import polars as pl

    fit_codes, valid_codes, cards = [], [], []
    for name in fit.columns:
        counts = fit.group_by(name).len().filter((pl.col("len") >= minimum) & pl.col(name).is_not_null())
        vocab = counts.sort(name).with_row_index("code", offset=1).select(name, pl.col("code").cast(pl.Int32))
        for frame, store in ((fit, fit_codes), (valid, valid_codes)):
            joined = frame.select(name).with_row_index("_i").join(vocab, on=name, how="left").sort("_i")
            store.append(joined["code"].fill_null(0).to_numpy())
        cards.append(len(vocab) + 1)
    return np.stack(fit_codes, 1).astype(np.int32), np.stack(valid_codes, 1).astype(np.int32), cards


def prepare(args):
    import polars as pl
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from xgb_tune import BASE, CONFIGS, train_split_columns

    m, out = args.matrices, args.output
    out.mkdir(parents=True, exist_ok=True)
    began = time.time()
    x_fit = np.load(m / "x_fit.npy", mmap_mode="r")
    x_valid = np.load(m / "x_valid.npy", mmap_mode="r")
    y_fit, y_valid = np.load(m / "y_fit.npy"), np.load(m / "y_valid.npy")
    grid = quantile_grid(x_fit)
    np.save(out / "bins_fit.npy", bin_matrix(x_fit, grid))
    np.save(out / "bins_valid.npy", bin_matrix(x_valid, grid))
    print(f"binned in {time.time() - began:.0f}s", flush=True)

    fit_ids, valid_ids = train_split_columns(args.data, ID_COLUMNS, args.rows)
    ids_fit, ids_valid, cards = id_codes(fit_ids, valid_ids)
    np.save(out / "ids_fit.npy", ids_fit)
    np.save(out / "ids_valid.npy", ids_valid)
    coverage = {c: float((ids_valid[:, i] > 0).mean()) for i, c in enumerate(ID_COLUMNS)}
    print("id cards", cards, "holdout coverage", coverage, flush=True)

    params = {**BASE, **CONFIGS["depth12_node"], "device": args.device}
    train = xgb.QuantileDMatrix(np.asarray(x_fit), y_fit, max_bin=params["max_bin"])
    valid = xgb.QuantileDMatrix(np.asarray(x_valid), y_valid, ref=train, max_bin=params["max_bin"])
    booster = xgb.train(params, train, args.trees, evals=[(valid, "holdout")], verbose_eval=250)
    leaves = {}
    for name, dm in (("fit", train), ("valid", valid)):
        leaves[name] = booster.predict(dm, pred_leaf=True).astype(np.int16)
        np.save(out / f"leaves_{name}.npy", leaves[name])
    auc = float(roc_auc_score(y_valid, booster.predict(valid)))
    leaf_nodes = int(max(leaves["fit"].max(), leaves["valid"].max())) + 1
    meta = {"id_columns": ID_COLUMNS, "id_cards": cards, "id_holdout_coverage": coverage,
            "n_dense": int(x_fit.shape[1]), "trees": args.trees, "leaf_nodes": leaf_nodes,
            "leaf_xgb_holdout_auc": auc, "seconds": time.time() - began}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
