Subject: ShareChat RecSys Challenge – Final Solution Writeup, Code & Artifacts (Team: dheerajgattupalli)

Hi Saira and the ShareChat ML team,

Thank you for organizing the ShareChat RecSys Challenge and for reaching out. Please find below our solution details for our **prize-eligible, strictly row-independent submission — S025 (Submission ID: `58816244`, Public ROC AUC: `0.83562`)**.

*(Disclosure on selected submissions: On Kaggle, we had selected `58816244` [S025, 0.83562, strictly row-independent] and `58479943` [S017, 0.88648, an earlier pre-clarification experiment that used `request_id` grouping]. Only `58816244` [S025] is our rule-compliant submission for prize review; all materials below correspond exclusively to S025.)*

1. **Detailed Writeup (including complete list of every evaluation feature):**
   - Located at `final_s025/SOLUTION_WRITEUP.md` in our repository: https://github.com/dgattupalli696/sharechat-recsys/blob/main/final_s025/SOLUTION_WRITEUP.md
   - Every feature used for test prediction is strictly row-independent (584 raw numeric scalars, 14×32 raw embeddings, masked low-cardinality categoricals, embedding cosines/dot+bias/norms/pairwise crosses, 36 duration-relative contrasts, 60 train-fitted paired-embedding rotations, 72 train-fitted match landmark distances, and 60 row-local rate/cosine/watch-time/age interactions). Zero `request_id` or cross-row features are used anywhere.

2. **Train vs. Test Feature Disclosure:**
   - Our approach uses the **exact same features** for both training and test. All scalers, quantile knots, categorical vocabularies, rotation matrices, and landmark centers are fitted strictly on `train.parquet` and frozen before scoring `test.parquet`. All ensemble combinations are pointwise per row (no test-set ranks or batch statistics).

3. **Full Code Repository:**
   - Repository: https://github.com/dgattupalli696/sharechat-recsys
   - Dedicated S025 review folder (writeup + frozen notebooks + assembly script + receipts): https://github.com/dgattupalli696/sharechat-recsys/tree/main/final_s025

4. **Trained Model Artifacts (all 19 trained models + frozen preprocessors):**
   - Attached to GitHub Release `s025-final` (`s025-trained-model-artifacts.zip`, 96.9 MiB, with SHA-256 `MANIFEST.json`):
   - https://github.com/dgattupalli696/sharechat-recsys/releases/tag/s025-final

5. **In-Person Finale & T-Shirt Size:**
   - **Ability to travel to ShareChat HQ, Bangalore if shortlisted:** `<Yes / No>`
   - **T-shirt size(s):** `<Your size, e.g., M / L / XL>`

Please let us know if you need any additional details or access permissions.

Best regards,
Dheeraj Gattupalli
Team: `dheerajgattupalli`
