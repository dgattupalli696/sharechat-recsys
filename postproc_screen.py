"""Holdout screen of post-processing transforms on the S017-equivalent blend: AUC and Spearman vs base."""
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

z = np.load("artifacts/local-t052/teacher-blend.npz")
ids, p = z["row_ids"], z["probabilities"].astype(np.float64)
rows = pl.scan_parquet("data/train.parquet").with_row_index("_row_id").filter(pl.col("_row_id").is_in(ids.tolist()))
f = rows.select("_row_id", "watched", "request_id", "feature1", "feature3", "feature5", "feature18", "feature23").collect()
f = f.join(pl.DataFrame({"_row_id": ids, "s": np.log(p / (1 - p))}), on="_row_id")
f = f.with_columns(pl.col("feature1").cast(pl.Int64).alias("t"))
f = f.with_columns((pl.col("t") // 3600000).alias("hourabs"), (pl.col("t") // 600000).alias("min10"))
y, base = f["watched"].to_numpy(), f["s"].to_numpy()
rb = rankdata(base)
print("base", round(roc_auc_score(y, base), 5), "rows", len(y), "t range", f["t"].min(), f["t"].max())


def report(name, score):
    print(f"{name:34s} auc {roc_auc_score(y, score):.5f}  d {roc_auc_score(y, score) - roc_auc_score(y, base):+.5f}"
          f"  spearman {np.corrcoef(rb, rankdata(score))[0, 1]:.5f}", flush=True)


for g in ["request_id", "feature5", "feature18", "feature23", "hourabs", "min10"]:
    m = f.select(pl.col("s").mean().over(g)).to_series().to_numpy()
    n = f.select(pl.len().over(g)).to_series().to_numpy()
    for b in [-0.3, -0.1, 0.1, 0.3]:
        report(f"{g} mean {b:+}", base + b * m)
    if g in ["request_id", "feature5"]:
        r = f.select(((pl.col("s").rank().over(g) - 0.5) / pl.len().over(g)) - 0.5).to_series().to_numpy()
        for a in [0.1, 0.3]:
            report(f"{g} within-rank {a:+}", rb / len(rb) + a * r * (n > 1))
