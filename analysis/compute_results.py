"""Re-aggregate model performance from saved predictions (no model retraining).

- DNN: 10 seeds (Phase 2E per-seed predictions); report mean +/- SD of per-seed metrics,
  95% CI = game-cluster bootstrap of the seed-mean metric.
- Comparison models: saved predictions (Phase 1E for 2025, Phase 2C for 2026).
  In-game logistic regression is refit with equal per-game weights.
- Baselines: home-rate constant (training-set home win rate, game level),
  empirical win expectancy and leading-team rule (in-game).
- External 2025->2026: frozen 2025 DNN (10 seeds), 2025-fit LR / home-rate / EWE / leading rule.
- Bootstrap: B = 2000, resampling games, percentile CIs, same draws for all models (paired).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

D = Path(sys.argv[1]); P2E = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
B = 2000; RNG_SEED = 2026

PRE = ['home_win_pct','away_win_pct','win_pct_diff','home_run_diff','away_run_diff','run_diff_diff',
       'home_runs_per_game','away_runs_per_game','home_ra_per_game','away_ra_per_game',
       'home_recent_form','away_recent_form','form_diff','home_sp_era','away_sp_era','sp_era_diff',
       'home_sp_whip','away_sp_whip','sp_whip_diff','home_sp_k9','away_sp_k9','sp_k9_diff']
ING = ['inning','is_bottom','score_diff','home_runs','away_runs','home_hits','away_hits','abs_score_diff']
KEY_IN = ['gamePk','inning','is_bottom']

# ---------------- data ----------------
pre25 = pd.read_csv(D/'phase1c_output/pregame_features_cleaned_2430.csv')
pre26 = pd.read_csv(D/'phase2c_2026_retraining_output/2026_pregame_features_training_cleaned.csv')
in25 = pd.read_csv(D/'phase1d_output/ingame_snapshots_2430.csv')
in26 = pd.read_csv(D/'phase2b_2026_external_output/2026_ingame_predictions.csv').drop(columns=['prob_ingame_dnn'])
in26 = in26.merge(pd.read_csv(D/'phase2c_2026_retraining_output/2026_game_partitions.csv')[['gamePk','split']], on='gamePk', validate='many_to_one')
ext_pre = pd.read_csv(D/'phase2b_2026_external_output/2026_pregame_features.csv')
med25 = json.load(open(D/'phase1c_output/training_audit.json'))['imputation']['pitcher_train_only_medians']
for side in ('home','away'):
    miss = ext_pre[f'{side}_sp_source'] != 'prior_game_logs'
    for st in ('era','whip','k9'): ext_pre.loc[miss, f'{side}_sp_{st}'] = med25[st]
for st in ('era','whip','k9'): ext_pre[f'sp_{st}_diff'] = ext_pre[f'home_sp_{st}'] - ext_pre[f'away_sp_{st}']

seedp = pd.read_csv(P2E/'per_seed_predictions.csv.gz')
SEEDS = sorted(seedp.seed.unique()); assert len(SEEDS) == 10

def game_w(d):
    w = 1.0 / d.groupby('gamePk').gamePk.transform('size').to_numpy(float); return w / w.mean()

def ewe_fit(train):
    w = game_w(train); t = train.assign(sb=train.score_diff.clip(-5,5), w=w, wy=w*train.target)
    prior = float(t.wy.sum()/t.w.sum())
    cells = t.groupby(['inning','is_bottom','sb']).agg(n=('w','sum'), wins=('wy','sum'))
    def f(d):
        k = pd.MultiIndex.from_arrays([d.inning.astype(int), d.is_bottom.astype(int), d.score_diff.clip(-5,5).astype(int)])
        c = cells.reindex(k); return ((c.wins.fillna(0)+30*prior)/(c.n.fillna(0)+30)).to_numpy()
    return f, prior

def lead_rule(d, prior):
    s = d.score_diff.to_numpy(); return np.where(s>0, 1., np.where(s<0, 0., prior))

def home_rate(train):  # game-level home win rate in training data
    return float(train.groupby('gamePk').target.first().mean())

def lr_fit(train, F, weighted):
    sc = StandardScaler().fit(train[F].to_numpy(float))
    w = game_w(train) if weighted else None
    m = LogisticRegression(C=1.0, max_iter=2500, random_state=2026).fit(sc.transform(train[F].to_numpy(float)), train.target, sample_weight=w)
    return lambda d: m.predict_proba(sc.transform(d[F].to_numpy(float)))[:,1]

# ---------------- assemble evaluation frames ----------------
def dnn_cols(task, ev, frame, keys):
    s = seedp[(seedp.task==task) & (seedp['eval']==ev)]
    wide = s.pivot_table(index=keys, columns='seed', values='prob').reset_index()
    wide.columns = keys + [f'DNN_s{c}' for c in wide.columns[len(keys):]]
    out = frame.merge(wide, on=keys, how='left', validate='one_to_one')
    assert out[[f'DNN_s{c}' for c in SEEDS]].notna().all().all(), (task, ev)
    return out

evals = {}
# Pre-game 2025 test
f = pd.read_csv(D/'phase1e_baseline_output/pregame_comparison_test_predictions.csv').rename(columns={'DNN':'DNN_saved'})
f['Home-rate baseline'] = home_rate(pre25[pre25.split=='train'])
evals[('Pre-game','2025 test')] = dnn_cols('Pre-game 2025','test', f, ['gamePk'])
# Pre-game 2026 test
f = pd.read_csv(D/'phase2c_2026_retraining_output/2026_pregame_test_predictions.csv').rename(columns={'DNN':'DNN_saved'})
f['Home-rate baseline'] = home_rate(pre26[pre26.split=='train'])
evals[('Pre-game','2026 test')] = dnn_cols('Pre-game 2026','test', f, ['gamePk'])
# Pre-game external
f = ext_pre[['gamePk','target']].copy()
f['Home-rate baseline'] = home_rate(pre25[pre25.split=='train'])
f['Logistic Regression'] = lr_fit(pre25[pre25.split=='train'], PRE, False)(ext_pre)
evals[('Pre-game','2025→2026 external')] = dnn_cols('Pre-game 2025','external_2026', f, ['gamePk'])

# In-game 2025 / 2026 test: saved trees/EWE/rule; LR refit with game weights
for yr, path, frame in [('2025', D/'phase1e_baseline_output/ingame_comparison_test_predictions.csv', in25),
                        ('2026', D/'phase2c_2026_retraining_output/2026_ingame_test_predictions.csv', in26)]:
    f = pd.read_csv(path).rename(columns={'DNN':'DNN_saved','Logistic Regression':'LR_unweighted_saved'})
    tr = frame[frame.split=='train']; te = frame[frame.split=='test']
    lrp = te[KEY_IN].assign(**{'Logistic Regression': lr_fit(tr, ING, True)(te)})
    f = f.merge(lrp, on=KEY_IN, validate='one_to_one')
    f['Home-rate baseline'] = home_rate(tr)
    task = f'In-game {yr}'
    evals[('In-game', f'{yr} test')] = dnn_cols(task, 'test', f, KEY_IN)
# In-game external
tr = in25[in25.split=='train']; ewe, prior = ewe_fit(tr)
f = in26[KEY_IN+['target','score_diff']].copy()
f['Home-rate baseline'] = home_rate(tr)
f['Logistic Regression'] = lr_fit(tr, ING, True)(in26)
f['Empirical Win Expectancy'] = ewe(in26); f['Leading-Team Rule'] = lead_rule(in26, prior)
evals[('In-game','2025→2026 external')] = dnn_cols('In-game 2025','external_2026', f, KEY_IN)

# ---------------- metrics ----------------
from scipy.stats import rankdata
def ece10(y, p):
    b = np.clip(np.digitize(p, np.linspace(0,1,11)[1:-1], right=True), 0, 9)
    n = np.bincount(b, minlength=10); sy = np.bincount(b, y, 10); sp = np.bincount(b, p, 10)
    m = n > 0
    return float(np.sum(np.abs(sy[m]-sp[m])) / len(y))
def auc(y, p):
    if np.all(p == p[0]): return 0.5
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y==1].sum() - n1*(n1+1)/2) / (n1*n0))
def metrics(y, p):
    pc = np.clip(p, 1e-7, 1-1e-7)
    return dict(accuracy=float(np.mean((p>=.5)==y)), auc=auc(y,p), brier=float(np.mean((p-y)**2)),
                logloss=float(-np.mean(y*np.log(pc)+(1-y)*np.log(1-pc))), ece10=ece10(y,p))

MODEL_ORDER = ['Home-rate baseline','Empirical Win Expectancy','Leading-Team Rule','Logistic Regression',
               'Random Forest','Histogram Gradient Boosting','XGBoost','DNN']
rows = []; diffs = []; checks = []; frames = {}
rng = np.random.default_rng(RNG_SEED)
for (task, ev), df in evals.items():
    df = df.reset_index(drop=True); y = df.target.to_numpy(int)
    games, inv = np.unique(df.gamePk.to_numpy(), return_inverse=True)
    order = np.argsort(inv, kind='stable'); starts = np.searchsorted(inv[order], np.arange(len(games)))
    ends = np.append(starts[1:], len(df)); rows_of = [order[a:b] for a, b in zip(starts, ends)]
    models = [m for m in MODEL_ORDER if m=='DNN' or m in df.columns]
    seedcols = [f'DNN_s{s}' for s in SEEDS]
    P = {m: df[m].to_numpy(float) for m in models if m!='DNN'}
    PS = [df[c].to_numpy(float) for c in seedcols]
    df['DNN'] = np.mean(PS, axis=0)  # seed-average prediction (for figures only)
    frames[(task, ev)] = df
    if 'DNN_saved' in df:
        checks.append(dict(task=task, eval=ev, max_abs_diff_saved_vs_seed2026=float(np.abs(df.DNN_saved-df['DNN_s2026']).max())))
    if 'LR_unweighted_saved' in df:
        checks.append(dict(task=task, eval=ev, lr_unweighted_brier=float(np.mean((df.LR_unweighted_saved-y)**2)),
                           lr_weighted_brier=float(np.mean((df['Logistic Regression']-y)**2))))
    def all_metrics(idx, keys=('accuracy','auc','brier','logloss','ece10')):
        yy = y[idx]; out = {}
        for m, p in P.items(): out[m] = metrics(yy, p[idx])
        ms = [metrics(yy, p[idx]) for p in PS]
        out['DNN'] = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        return out
    full = all_metrics(np.arange(len(df)))
    per_seed = [metrics(y, p) for p in PS]
    boot = {m: {k: np.empty(B) for k in ('accuracy','auc','brier')} for m in models}
    bdiff = {m: np.empty(B) for m in models if m!='DNN'}
    for b in range(B):
        gs = rng.integers(0, len(games), len(games))
        idx = np.concatenate([rows_of[g] for g in gs])
        bm = all_metrics(idx)
        for m in models:
            for k in boot[m]: boot[m][k][b] = bm[m][k]
        for m in bdiff: bdiff[m][b] = bm['DNN']['brier'] - bm[m]['brier']
    for m in models:
        r = dict(task=task, eval=ev, model=m, n_games=len(games), n_rows=len(df), **full[m])
        for k in ('accuracy','auc','brier'):
            r[f'{k}_lo'], r[f'{k}_hi'] = np.percentile(boot[m][k], [2.5, 97.5])
        if m == 'DNN':
            for k in ('accuracy','auc','brier','logloss','ece10'): r[f'{k}_sd'] = float(np.std([x[k] for x in per_seed], ddof=1))
        rows.append(r)
    for m, v in bdiff.items():
        diffs.append(dict(task=task, eval=ev, contrast=f'DNN - {m}', brier_diff=full['DNN']['brier']-full[m]['brier'],
                          lo=float(np.percentile(v,2.5)), hi=float(np.percentile(v,97.5))))
    print('done', task, ev, len(df), flush=True)

res = pd.DataFrame(rows); res.to_csv(OUT/'results_main.csv', index=False)
pd.DataFrame(diffs).to_csv(OUT/'results_brier_differences.csv', index=False)
pd.DataFrame(checks).to_csv(OUT/'sanity_checks.csv', index=False)

# ---------------- inning table (test sets) ----------------
inn = []
for (task, ev), df in frames.items():
    if task != 'In-game': continue
    g = df.assign(inn=np.where(df.inning>=10, '10+', df.inning.astype(str)))
    for k, d in g.groupby('inn', sort=False):
        yy = d.target.to_numpy(int)
        r = dict(eval=ev, inning=k, n=len(d))
        for m in ['DNN','Empirical Win Expectancy','Leading-Team Rule','Logistic Regression']:
            if m == 'DNN':
                ms = [metrics(yy, d[f'DNN_s{s}'].to_numpy()) for s in SEEDS]
                r['DNN_acc'] = np.mean([x['accuracy'] for x in ms]); r['DNN_brier'] = np.mean([x['brier'] for x in ms])
            else:
                mm = metrics(yy, d[m].to_numpy()); r[f'{m}_acc'] = mm['accuracy']; r[f'{m}_brier'] = mm['brier']
        inn.append(r)
inn = pd.DataFrame(inn); inn['ord'] = inn.inning.map(lambda x: 10 if x=='10+' else int(x))
inn.sort_values(['eval','ord']).drop(columns='ord').to_csv(OUT/'results_by_inning.csv', index=False)

# save seed-averaged prediction frames for figures
for (task, ev), df in frames.items():
    keep = [c for c in df.columns if c in ['gamePk','inning','is_bottom','target','DNN'] + MODEL_ORDER]
    df[keep].to_csv(OUT/f"pred_{task.replace('-','').replace(' ','')}_{ev.replace(' ','_').replace('→','to')}.csv", index=False)
print('ALL DONE')
