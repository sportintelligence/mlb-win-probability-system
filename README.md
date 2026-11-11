# Real-Time MLB Win Probability Prediction System

Code and trained models for a browser-based MLB win probability dashboard that provides pre-game and in-game (half-inning) win probabilities from the MLB Stats API.

Web demo: https://sportintelligence.github.io/mlb-win-probability-system/

## Contents

| Path | Contents |
| --- | --- |
| `index.html` | Dashboard. Trained DNN weights are embedded as JSON and inference runs in the browser (no server needed). |
| `notebooks/` | Data collection, feature construction, model training and evaluation (Google Colab), run in numbered order |
| `analysis/` | Scripts that summarize model performance, SHAP and calibration from the notebook outputs |
| `models/` | Trained pre-game and in-game DNNs (`.keras`) and their JSON exports |

## Models

- Pre-game DNN: 22 inputs from team and starting-pitcher records dated before the game; Input(22) → 128 → 64 → 32 → 1.
- In-game DNN: 8 game-state inputs at the start of each half-inning; Input(8) → 64 → 32 → 16 → 1.
- The dashboard uses the models trained on the 2025 regular season (random seed 2026).

## Usage

Open `index.html` in a browser, or use the web demo. Results depend on MLB Stats API availability.

## Data

Game data come from the MLB Stats API (statsapi.mlb.com) and are subject to MLB Advanced Media's terms of use. They are used here for non-commercial academic research. Raw API data are not redistributed; the notebooks collect them directly.

## License

MIT License (see `LICENSE`).
