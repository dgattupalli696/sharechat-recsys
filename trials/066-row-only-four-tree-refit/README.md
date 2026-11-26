# T066 – full-data refit of frozen T065 winner

Fit missing depth12-rich, depth6-original, depth6-rich on all 1,537,638 labels, unchanged T063 seed 42 / 900 rounds. Stream unchanged same-row transforms in file order. Reuse T062 original depth12 and T021 full neural/tree ensemble. Frozen pointwise weights 50% T021 and 12.5% each of four XGBs. Full refit carries no independent AUC. Reproduce source/notebook with build.py, then assemble.py on downloaded t066 directory. No competition upload.
