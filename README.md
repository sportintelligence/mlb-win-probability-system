# Real-Time MLB Win Probability Prediction System

Code, models and results for the manuscript
"A Deep Learning-Based Real-Time MLB Game Outcome Prediction System Using a Public Sports API: Design and Implementation" (under review).

Web demo: https://sportintelligence.github.io/mlb-win-probability-system/

## What the study found

- In-game win probability is estimated stably from a small set of game-state variables (inning, half, score, hits). The DNN's Brier score was 0.003–0.004 lower than an empirical win-expectancy table, and the gap held when the 2025 model was applied unchanged to the 2026 season.
- Pre-game prediction from prior-date team and starting-pitcher statistics barely exceeded a constant home-win-rate baseline within a season, and the gain did not carry over to 2026.

## Dashboard (`index.html`)

A single HTML file. Open it in a browser or use the web demo. No server is needed: the trained weights are embedded as JSON and inference runs in JavaScript.

- Pre-game DNN: Input(22) → 128 → 64 → 32 → 1, trained on the 2025 season, seed 2026.
- In-game DNN: Input(8) → 64 → 32 → 16 → 1, trained on the 2025 season, seed 2026. One prediction at the start of each half-inning.
- Data: MLB Stats API (`schedule`, `people/{id}/stats`, `game/{pk}/linescore`, `game/{pk}/boxscore`, `game/{pk}/feed/live`), refreshed every 30 seconds.
- Pre-game inputs use only records dated before the game. For 2025 games, the inputs from the training pipeline are embedded in the page; for other games they are built from API responses.
- Predicted probabilities are shown as computed (no clipping).
- The comparison tables show 2025 test-set performance. The DNN rows are the deployed seed-2026 model; the manuscript tables report the mean of 10 seeds.

## Repository layout

| Path | Contents |
| --- | --- |
| `index.html` | Dashboard |
| `notebooks/` | Analysis pipeline, run in order on Google Colab (see below) |
| `analysis/` | Scripts that re-aggregate saved predictions into the manuscript tables and figures |
| `web_verification/` | Python–JavaScript prediction parity check and JavaScript timing |
| `models/` | Trained DNNs (`.keras`) and their JSON exports (weights, scaler, feature order) |
| `results/` | Result tables used in the manuscript |

## Notebooks

| Notebook | Step |
| --- | --- |
| `01_Phase1A_Sample_Audit` | Regular-season sample (gameType = R, completed games, 2,430 games) |
| `02_Phase1B_PreGame_Feature_Audit` | Prior-date pre-game features (22 variables) |
| `03_Phase1C_PreGame_Training` | Chronological split and pre-game DNN training |
| `04_Phase1C2_DNN_Architecture_Tuning` | Pre-game DNN candidate selection by validation Brier score |
| `05_Phase1D_InGame_Training` | Half-inning snapshots and in-game DNN training |
| `06_Phase1E_Baseline_Comparison` | Logistic regression, random forest, gradient boosting, XGBoost, baselines |
| `07_Phase2A_Sensitivity_and_SHAP` | Sensitivity analyses and SHAP |
| `08_Phase2B_2026_External_Validation` | 2025 models applied unchanged to the 2026 season |
| `09_Phase2C_2026_Independent_Retraining` | Same protocol retrained on 2026 |
| `10_Phase2E_Seed_Robustness` | DNN retrained with 10 seeds (2026–2035) |

Each notebook reads the outputs of the earlier steps (uploaded in Colab) and collects game data directly from the MLB Stats API. Raw API data are not redistributed here; rerunning notebooks 01, 02, 05 and 08 recollects them.

## Analysis scripts

```
python analysis/compute_results.py <phase_outputs_dir> <phase2e_output_dir> <out_dir>   # Tables 6–7, Brier differences, by-inning results
python analysis/grouped_shap.py    <phase_outputs_dir> <out_dir>                         # individual and grouped SHAP
python analysis/figure2_shap.py    <shap_dir> <out_dir>                                  # Figure 2
python analysis/calibration.py     <phase2e_output_dir> <out_dir>                         # Figure 4, ECE
node   web_verification/verify_js_parity.js <parity_input.json>                         # Keras vs JavaScript predictions
node   web_verification/time_forward.js     <parity_input.json>                         # JavaScript forward-pass time
```

`<phase_outputs_dir>` holds the notebook output folders under their default names (`phase1c_output`, `phase1c2_output`, `phase1d_output`, `phase1e_baseline_output`, `phase2b_2026_external_output`, `phase2c_2026_retraining_output`). Confidence intervals use 2,000 game-level bootstrap resamples.

## Data and terms of use

Game data come from the MLB Stats API (statsapi.mlb.com), which is subject to MLB Advanced Media's terms of use. This repository uses the data for non-commercial academic research only and does not redistribute raw API data.

## License

MIT License (see `LICENSE`).
