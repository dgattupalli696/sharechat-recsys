# T023 - Request context, train+test counts, target encoding (local LightGBM screen)

Train on rows before the original primary cutoff (1,230,110); score the original
307,528-row holdout. Scalar features only (list/vector columns excluded),
LightGBM lr 0.05, 127 leaves, early stopping on the holdout (same for every
variant, so absolute AUCs are mildly optimistic; deltas are comparable).
`request_id` is used only for grouping, never as an input.

| Variant | Holdout AUC | Trees |
| --- | ---: | ---: |
| Base scalar features | 0.828722 | 117 |
| + Request context (size, distinct items, within-request rank/deviation of top features) | **0.882612** | 198 |
| + Label-free train+test counts (user/item/entity counts, gaps, ages) | 0.878542 | 185 |
| + Grouped out-of-fold target encoding | 0.828468 | 131 |
| All three | 0.882526 | 191 |

Request context gives +0.054 over base and beats the T016 neural ensemble
(0.840806). Counts largely overlap with it (`cnt_feature5` mostly tracks
request size because most users have one request). Target encoding adds
nothing. Larger tree budgets are not the bottleneck: early stopping picks
~120-200 trees.

Profiling: `feature5` behaves like a user ID (constant within a request; 7% of
test users seen in train); `feature18`/`feature24` like an item ID. Every
request has one timestamp; request IDs never cross train/holdout/test; request
size distributions match between train and test.
