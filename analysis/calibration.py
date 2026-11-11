"""Figure 4. Calibration curves for the DNN (10-seed mean prediction). No retraining.
Bins: equal-frequency (quantile) bins, 5 for pre-game, 10 for in-game. Pre-game error bars: 95% binomial CI. ECE = sum_b (n_b/N)|mean(y)_b - mean(p)_b|,
computed per seed with that seed's quantile bins and averaged over 10 seeds (SD reported)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Usage: python calibration.py <phase2e_output_dir> <out_dir>
P2E, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
seedp = pd.read_csv(P2E/'per_seed_predictions.csv.gz')
SETS = {'Pre-game': [('Pre-game 2025','test','2025 test'), ('Pre-game 2026','test','2026 test'),
                     ('Pre-game 2025','external_2026','2025→2026 external')],
        'In-game':  [('In-game 2025','test','2025 test'), ('In-game 2026','test','2026 test'),
                     ('In-game 2025','external_2026','2025→2026 external')]}
NB = 10
def qbins(p, NB):
    edges = np.quantile(p, np.linspace(0, 1, NB+1)); edges[0], edges[-1] = -np.inf, np.inf
    return np.clip(np.searchsorted(edges, p, side='right') - 1, 0, NB-1)
def ece_q(y, p, NB):
    b = qbins(p, NB); n = np.bincount(b, minlength=NB); m = n > 0
    return float(np.sum(np.abs(np.bincount(b, y, NB)[m] - np.bincount(b, p, NB)[m])) / len(y))
def curve(y, p, NB):
    b = qbins(p, NB); n = np.bincount(b, minlength=NB); m = n > 0
    o = np.bincount(b, y, NB)[m]/n[m]
    return np.bincount(b, p, NB)[m]/n[m], o, 1.96*np.sqrt(o*(1-o)/n[m])

rows, curves = [], {}
for task, sets in SETS.items():
    nb = 5 if task=='Pre-game' else 10
    for t, ev, lab in sets:
        s = seedp[(seedp.task==t) & (seedp['eval']==ev)]
        keys = ['gamePk'] if task=='Pre-game' else ['gamePk','inning','is_bottom']
        w = s.pivot_table(index=keys+['target'], columns='seed', values='prob').reset_index()
        y = w.target.to_numpy(float); P = w[sorted(seedp.seed.unique())].to_numpy(float)
        e = [ece_q(y, P[:, k], nb) for k in range(P.shape[1])]
        pm = P.mean(1)
        rows.append(dict(task=task, bins=nb, set=lab, n=len(y), ece_mean=np.mean(e), ece_sd=np.std(e, ddof=1),
                         ece_seedmean_pred=ece_q(y, pm, nb), p_min=pm.min(), p_p05=np.quantile(pm,.05),
                         p_p95=np.quantile(pm,.95), p_max=pm.max()))
        curves[(task, lab)] = curve(y, pm, nb)
tab = pd.DataFrame(rows); tab.to_csv(OUT/'calibration_ece_quantile.csv', index=False)
print(tab.round(4).to_string(index=False))

plt.rcParams.update({'font.family': 'Inter', 'font.size': 11})
COL = {'2025 test': ('#0B2D6B','o','-'), '2026 test': ('#5BA8D9','s','--'), '2025→2026 external': ('#E8590C','^','-.')}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, (task, lim, ttl) in zip(axes, [('Pre-game',(0.2,0.8),'(a) Pre-game model'), ('In-game',(0,1),'(b) In-game model')]):
    ax.plot(lim, lim, color='#999', lw=1, ls=':', zorder=1, label='Perfect calibration')
    for lab in COL:
        x, yb, ci = curves[(task, lab)]; c, mk, ls = COL[lab]
        if task == 'Pre-game': ax.errorbar(x, yb, yerr=ci, fmt='none', ecolor=c, elinewidth=1, capsize=3, alpha=.8, zorder=2)
        r = tab[(tab.task==task)&(tab.set==lab)].iloc[0]
        ax.plot(x, yb, color=c, marker=mk, ls=ls, lw=1.6, ms=6, label=f'{lab} (ECE = {r.ece_mean:.3f})', zorder=3)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Predicted home win probability (bin mean)'); ax.set_title(ttl, loc='left')
    ax.grid(alpha=.3); ax.legend(loc='upper left', fontsize=9.5, frameon=False)
axes[0].set_ylabel('Observed home win rate')
fig.tight_layout()
fig.savefig(OUT/'Figure4_calibration.png', dpi=300, bbox_inches='tight')
