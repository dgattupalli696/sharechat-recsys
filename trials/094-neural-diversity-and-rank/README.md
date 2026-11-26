# T094 – Enriched neural diversity (DCNv2-r64, DCNv2-MoE4, FinalMLP) + early-peaking global rank:pairwise XGBoost

Combine the two most orthogonal leads from T090 and T093 on the 1,240 super-enriched row-only features across both chronological folds:
1. `dcnv2_r64`: 3-layer DCNv2 (rank 64, width 256, PLR + pairwise vector crosses + low-card cats), epochs 1–4.
2. `dcnv2_moe4`: 2-layer 4-expert MoE DCNv2 (rank 32, width 256, PLR + vector crosses + low-card cats), epochs 1–4.
3. `finalmlp`: dual-stream gated bilinear FinalMLP (width 256, PLR + vector crosses + low-card cats), epochs 2/4/6.
4. `rank_d5` & `rank_d6`: global `objective='rank:pairwise'` XGBoost at early checkpoints (300/500/700/900 rounds) before rank-loss overfitting.
Strictly row-independent; zero `request_id` usage; singleton invariance verified. Joint Caruana logit ensemble selected on earlier fold and evaluated frozen on primary. No submission.
