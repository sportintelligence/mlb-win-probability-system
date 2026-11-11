"""Figure 2. Mean absolute SHAP values: (a) 22 individual predictors, (b) four predictor groups."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import cm, colors
import sys
# Usage: python figure2_shap.py <shap_dir> <out_dir>  (inputs from grouped_shap.py)
SH, OUT = sys.argv[1].rstrip('/')+'/', sys.argv[2].rstrip('/')+'/'
ind=pd.read_csv(SH+'shap_22_individual.csv').sort_values('mean_abs_shap')
grp=pd.read_csv(SH+'shap_22_grouped4.csv').sort_values('mean_abs_group_shap')
plt.rcParams.update({'font.family':'Inter','font.size':10.5})
cmap=matplotlib.colormaps['Blues']
def col(v,vmax): return cmap(0.25+0.75*v/vmax)
fig,(a1,a2)=plt.subplots(2,1,figsize=(7.5,9.6),gridspec_kw={'height_ratios':[22,6.5]})
a1.barh(ind.feature,ind.mean_abs_shap,color=[col(v,ind.mean_abs_shap.max()) for v in ind.mean_abs_shap],height=.72)
a1.set_title('(a) Individual predictors (22)',loc='left'); a1.set_ylim(-.6,len(ind)-.4)
a1.set_xlabel('Mean absolute SHAP value (home-win probability)')
lab=[f'{g} ({n})' for g,n in zip(grp.group,grp.n_vars)]
a2.barh(lab,grp.mean_abs_group_shap,color=[col(v,grp.mean_abs_group_shap.max()) for v in grp.mean_abs_group_shap],height=.62)
for i,v in enumerate(grp.mean_abs_group_shap): a2.text(v+.0002,i,f'{v:.3f}',va='center',fontsize=10)
a2.set_xlim(0,grp.mean_abs_group_shap.max()*1.12)
a2.set_title('(b) Predictor groups (4)',loc='left')
a2.set_xlabel('Mean absolute grouped SHAP value (home-win probability)')
fig.tight_layout(h_pad=1.6)
fig.savefig(OUT+'Figure2_SHAP.png',dpi=300,bbox_inches='tight')
