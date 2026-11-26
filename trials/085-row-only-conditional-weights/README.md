# T085 – row-local conditional blend weights

Keep T081 and T083 components fixed. Compare constant paired-tree weight against two binary routing rules: duration <=60 seconds or presence of feature32. Independently set both weights from 0, .05, .10, .167, .25 on the earlier fold; global AUC selects one rule including constant controls. Apply frozen on primary, no request inputs. Same gate as T084: improve both folds vs T083 and primary gain >=0.0003 vs S023. No new fits or leaderboard upload. Results in results.json.
