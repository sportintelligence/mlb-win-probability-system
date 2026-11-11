"""Individual and grouped SHAP for the deployed 22-feature pre-game DNN (2025 season, seed 2026). No retraining.
Usage: python grouped_shap.py <data_dir> <out_dir>
Settings follow Phase 2A: GradientExplainer, background 100 training games, explanation 150 test games,
rng = default_rng(2026), nsamples=100, rseed=2026.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf, shap

D = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
SEED = 2026
df = pd.read_csv(D/'phase1c_output/pregame_features_cleaned_2430.csv')
exp = json.load(open(D/'phase1c2_output/selected_pregame_export.json'))
F = exp['features']; mean = np.array(exp['scaler']['mean']); scale = np.array(exp['scaler']['scale'])
model = tf.keras.models.load_model(D/'phase1c2_output/selected_pregame_model.keras')
tr = df[df.split=='train']; te = df[df.split=='test']
X = lambda d: ((d[F].to_numpy(float)-mean)/scale).astype(np.float32)
# sanity: model reproduces saved test predictions
saved = pd.read_csv(D/'phase1c2_output/selected_test_predictions.csv')
p = model.predict(X(te), verbose=0).ravel()
print('max |pred - saved|:', float(np.abs(p - saved.set_index('gamePk').loc[te.gamePk, 'prob_selected_dnn'].to_numpy()).max()))

rng = np.random.default_rng(SEED)
bg = tr.iloc[rng.choice(len(tr), 100, replace=False)]
ex = te.iloc[rng.choice(len(te), 150, replace=False)]
expl = shap.GradientExplainer(model, X(bg))
sv = np.asarray(expl.shap_values(X(ex), nsamples=100, rseed=SEED))
sv = sv.reshape(sv.shape[0], -1) if sv.ndim == 3 and sv.shape[-1] == 1 else (sv[0] if sv.ndim == 3 else sv)
assert sv.shape == (150, 22), sv.shape
ind = pd.DataFrame({'feature': F, 'mean_abs_shap': np.abs(sv).mean(0), 'mean_shap': sv.mean(0)})

# Four predictor groups. Pitcher ERA, WHIP and K/9 are pooled because ERA and WHIP are highly
# correlated (r = 0.85) and split their attributions when kept apart.
GROUPS = {
    'Runs scored and allowed': ['home_runs_per_game','away_runs_per_game','home_ra_per_game','away_ra_per_game','home_run_diff','away_run_diff','run_diff_diff'],
    'Recent form': ['home_recent_form','away_recent_form','form_diff'],
    'Starting pitcher': ['home_sp_era','away_sp_era','sp_era_diff','home_sp_whip','away_sp_whip','sp_whip_diff','home_sp_k9','away_sp_k9','sp_k9_diff'],
    'Season winning percentage': ['home_win_pct','away_win_pct','win_pct_diff'],
}
# Direction: correlation between the group's summed SHAP value and the sum of its standardized
# home-minus-away difference variables, signed so that a larger value favours the home team.
DIFF_SIGN = {'Runs scored and allowed': {'run_diff_diff': 1}, 'Recent form': {'form_diff': 1},
             'Starting pitcher': {'sp_era_diff': -1, 'sp_whip_diff': -1, 'sp_k9_diff': 1},
             'Season winning percentage': {'win_pct_diff': 1}}
# Expected sign of each individual predictor on the home-win probability.
EXPECTED = {f: (-1 if f.startswith('away_') else 1) for f in F}
for f in ['home_ra_per_game','home_sp_era','home_sp_whip','sp_era_diff','sp_whip_diff']: EXPECTED[f] = -1
for f in ['away_ra_per_game','away_sp_era','away_sp_whip']: EXPECTED[f] = 1
assert sorted(sum(GROUPS.values(), [])) == sorted(F)
idx = {f: i for i, f in enumerate(F)}
Xex, Zex = ex[F].to_numpy(float), X(ex)
ind['direction_corr'] = [np.corrcoef(Xex[:, i], sv[:, i])[0, 1] for i in range(len(F))]
ind['expected_sign'] = [EXPECTED[f] for f in F]
ind['expected_direction'] = np.sign(ind.direction_corr) == ind.expected_sign
rows = []
for g, fs in GROUPS.items():
    s = sv[:, [idx[f] for f in fs]].sum(1)
    oriented = sum(sg * Zex[:, idx[f]] for f, sg in DIFF_SIGN[g].items())
    rows.append(dict(group=g, n_vars=len(fs), mean_abs_group_shap=float(np.abs(s).mean()),
                     direction_corr=float(np.corrcoef(oriented, s)[0, 1])))
grp = pd.DataFrame(rows).sort_values('mean_abs_group_shap', ascending=False)
ind.to_csv(OUT/'shap_22_individual.csv', index=False)
grp.to_csv(OUT/'shap_22_grouped4.csv', index=False)
np.save(OUT/'shap_22_values.npy', sv)
print(ind.sort_values('mean_abs_shap', ascending=False).round(4).to_string(index=False))
print(grp.round(4).to_string(index=False))
