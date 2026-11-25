import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--input',default='data/bootstrap_61.npz');p.add_argument('--out',default='results/bootstrap_map');args=p.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
z=np.load(args.input);orig=z['original'];bs=z['bootstrap']; grid=z['grid'];xg=z['xgrid'];yg=z['ygrid']; G=len(grid)
valid=np.isfinite(bs[:,2,:]).all(axis=1)
bs=bs[valid]
B=len(bs)
print('B',B,'min replicates cell',np.isfinite(bs).sum(axis=0).min())
# 95% percentile bootstrap interval excludes zero; pointwise non-adjusted
q_lo=np.nanquantile(bs,.025,axis=0);q_hi=np.nanquantile(bs,.975,axis=0)
label=np.where(q_lo>0,1,np.where(q_hi<0,-1,0))
# fallback for estimate missing
label[~np.isfinite(orig)]=0
series=['LLB-LD: original-scale local dependence', 'LLB-LD: copula-space local dependence', 'LF: copula-space relative local dependence']
files=['llb_ld_original_map.png','llb_ld_copula_map.png','lf_rld_map.png']
colors=ListedColormap(['#3062b0','#f5f4f0','#d34b42'])
norm=BoundaryNorm([-1.5,-.5,.5,1.5],colors.N)
for m in range(3):
    a=label[m].reshape(G,G)
    ext=[xg[0],xg[-1],yg[0],yg[-1]] if m==0 else [grid[0],grid[-1],grid[0],grid[-1]]
    fig,ax=plt.subplots(figsize=(7,5.7))
    im=ax.imshow(a,origin='lower',extent=ext,aspect='auto',cmap=colors,norm=norm,interpolation='nearest')
    ax.set_xlabel('x' if m==0 else 'u');ax.set_ylabel('y' if m==0 else 'v')
    ax.set_title(series[m]+f'\nPointwise 95% bootstrap CI, B={B}, n=1200')
    cb=fig.colorbar(im,ax=ax,ticks=[-1,0,1],shrink=.88);cb.ax.set_yticklabels(['Negative','Not significant','Positive'])
    fig.tight_layout();fig.savefig(out/files[m],dpi=195);plt.close(fig)
# put aligned maps in copula coords for easy comparison via xg yg index
fig,axs=plt.subplots(1,3,figsize=(14,4.3),sharex=True,sharey=True)
for m,ax in enumerate(axs):
    im=ax.imshow(label[m].reshape(G,G),origin='lower',extent=[grid[0],grid[-1],grid[0],grid[-1]],aspect='equal',cmap=colors,norm=norm,interpolation='nearest')
    ax.set_xlabel('Marginal rank u');ax.set_title(['LLB-LD (raw X,Y)','LLB-LD (rank U,V)','LF (rank U,V)'][m]);
    if m==0:ax.set_ylabel('Marginal rank v')
fig.colorbar(im,ax=axs.tolist(),ticks=[-1,0,1],fraction=.022,pad=.03).ax.set_yticklabels(['−','0','+'])
fig.suptitle(f'Bootstrap dependence maps | n=1200, h=0.25, B={B} | pointwise 95% intervals',fontsize=12)
fig.subplots_adjust(left=.07,right=.90,bottom=.12,top=.83,wspace=.14)
fig.savefig(out/'comparison_maps.png',dpi=195);plt.close(fig)
rows=[]
for k,(j,i) in enumerate([(j,i) for j in range(G) for i in range(G)]):
    row=dict(u=grid[i],v=grid[j],x=xg[i],y=yg[j])
    for m,name in enumerate(['LLB_LD_raw','LLB_LD_copula','LF_RLD']):
        row.update({f'{name}_estimate':orig[m,k],f'{name}_CI_lower':q_lo[m,k],f'{name}_CI_upper':q_hi[m,k],f'{name}_class':label[m,k]})
    rows.append(row)
pd.DataFrame(rows).to_csv(out/'bootstrap_map_grid.csv',index=False)
summary=[]
for m,name in enumerate(['LLB_LD_original','LLB_LD_copula','LF_RLD']):
  a=label[m]
  summary.append(dict(method=name,bootstrap_reps=B,grid=G,negative=int((a==-1).sum()),nonsignificant=int((a==0).sum()),positive=int((a==1).sum()),estimate_min=float(np.nanmin(orig[m])),estimate_max=float(np.nanmax(orig[m]))))
pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
