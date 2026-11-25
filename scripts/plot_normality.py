"""Regenerate normal QQ plots using archived simulation results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import probplot
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/normality';OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(ROOT/'data/normality_all_samples.csv')
for fam in ['Frank(3)','Clayton(1)','Gaussian(0.3)']:
 fig,axs=plt.subplots(3,4,figsize=(14,10),sharex=True,sharey=True)
 for a,method in enumerate(['LKT','LLB','LF']):
  for b,n in enumerate([4800,30000,100000,400000]):
   ax=axs[a,b];g=df[(df.family==fam)&(df.n==n)]
   z=(g[method]-g[method].mean())/g[method].std(ddof=1)
   q=probplot(z,dist='norm')
   ax.scatter(q[0][0],q[0][1],s=13,alpha=.6)
   ax.plot([-3,3],[-3,3],'k--',lw=.9)
   ax.set_xlim(-3.3,3.3);ax.set_ylim(-3.3,3.3)
   ax.set_title(f'{method}  n={n:,}',fontsize=11)
   if a==2:ax.set_xlabel('Theoretical normal quantile')
   if b==0:ax.set_ylabel('Empirical quantile')
 fig.suptitle(f'Normal Q-Q plots: {fam}, center (0.5,0.5)',fontsize=15)
 fig.tight_layout(rect=(0,0,1,.97))
 fig.savefig(OUT/('qq_'+fam.split('(')[0].lower()+'.png'),dpi=170)
 plt.close(fig)
print('QQ plots written to',OUT)
