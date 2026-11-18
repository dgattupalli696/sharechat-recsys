"""Profile every column: entity level (constant within item/user/request), range, timestamp signature,
label association, train→test drift and correlation blocks. Writes artifacts/local-column-profile/."""
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score

USER, ITEM = "feature5", "feature18"
OUT = Path("artifacts/local-column-profile")
schema = pl.scan_parquet("data/train.parquet").collect_schema()
tr = pl.scan_parquet("data/train.parquet").filter(pl.col("request_id").hash(11) % 5 == 0).collect()
te = pl.scan_parquet("data/test.parquet").filter(pl.col("request_id").hash(11) % 5 == 0).collect()
print(tr.shape, te.shape, flush=True)
lists = [c for c, t in schema.items() if str(t).startswith("List")]
scalars = [c for c in schema if c not in lists and c not in ("watched", "request_id", "feature1")]
y = tr["watched"].to_numpy()

def const_frac(df, key, cols):
    g = df.group_by(key).agg(pl.len().alias("_n"), *[pl.col(c).n_unique().alias(c) for c in cols if c != key]).filter(pl.col("_n") >= 2)
    return {c: (1.0 if c == key else float((g[c] == 1).mean())) for c in cols}

levels = {k: const_frac(tr, key, scalars) for k, key in [("item", ITEM), ("user", USER), ("request", "request_id")]}
print("levels done", flush=True)
rows = []
for c in scalars:
    s, t = tr[c], te[c] if c in te.columns else None
    r = {"col": c, "dtype": str(schema[c]), "null": round(s.null_count() / len(s), 3), "n_unique": s.n_unique(),
         **{f"const_{k}": round(v[c], 3) for k, v in levels.items()}}
    if s.dtype.is_numeric():
        v = s.cast(pl.Float64)
        q = [v.quantile(p) for p in (0, .01, .5, .99, 1)]
        r.update(min=q[0], p01=q[1], med=q[2], p99=q[3], max=q[4],
                 int_like=bool((v.drop_nulls().head(5000) % 1 == 0).all()),
                 unit01=bool(q[0] is not None and q[0] >= 0 and q[4] <= 1),
                 ts=("epoch_s" if q[2] and 1.6e9 < q[2] < 1.9e9 else "epoch_ms" if q[2] and 1.6e12 < q[2] < 1.9e12 else ""))
        x = v.fill_null(np.nan).to_numpy()
        ok = np.isfinite(x)
        if ok.sum() > 1000 and np.unique(x[ok][:20000]).size > 1:
            a = roc_auc_score(y[ok], x[ok]); r["auc"] = round(a, 4)
        if t is not None:
            tm = t.cast(pl.Float64).median()
            r["test_med_ratio"] = round(tm / q[2], 3) if q[2] not in (None, 0) and tm is not None else None
            r["test_null"] = round(t.null_count() / len(t), 3)
    else:
        top = tr.group_by(c).agg(pl.col("watched").mean().alias("w"), pl.len().alias("n")).filter(pl.col("n") >= 50)
        r["rate_spread"] = round(float(top["w"].std() or 0), 4)
        if t is not None:
            r["test_unseen"] = round(float((~te[c].is_in(tr[c].unique().implode())).mean()), 3)
    rows.append(r)
for c in lists:
    ln = tr[c].list.len()
    rows.append({"col": c, "dtype": "list", "null": round(tr[c].null_count() / len(tr), 3), "list_len_med": ln.median(),
                 "list_len_max": ln.max(), **{f"const_{k}": round(v, 3) for k, v in
                 {k: float(tr.group_by(key).agg(pl.len().alias("_n"), pl.col(c).list.sum().n_unique().alias("u")).filter(pl.col("_n") >= 2)["u"].eq(1).mean())
                  for k, key in [("item", ITEM), ("user", USER), ("request", "request_id")]}.items()}})
prof = pl.DataFrame(rows, infer_schema_length=None)
num = [r["col"] for r in rows if r.get("auc") is not None]
m = tr.select(num).sample(60000, seed=0).with_columns(pl.all().cast(pl.Float64).rank()).fill_null(0).to_numpy()
cc = np.nan_to_num(np.corrcoef(m.T))
Z = linkage(squareform(1 - np.abs(cc), checks=False), "average")
cl = fcluster(Z, 0.1, "distance")
prof = prof.join(pl.DataFrame({"col": num, "block": cl.astype(int)}), on="col", how="left")
prof.write_csv(OUT / "profile.csv")
print("written", prof.shape, flush=True)
