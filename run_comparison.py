"""Standalone paired Monte Carlo study: local Frank vs local log-bilinear.

Runs without prior CSV files. Both use rectangle kernels and raw copula uniforms.
Usage: python run_comparison.py --output results
Smoke test: python run_comparison.py --reps 1 --families 'Frank(3)' --bandwidths 0.25 --output smoke
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from relative_local_dependence.local_frank_fit import local_frank_fit
from relative_local_dependence.jones_koch_moment import jk_moment_rld
from relative_local_dependence.local_frank_spatial import fam_sample, fam_target

FAMILIES=['Frank(3)','Clayton(1)','Gaussian(0.3)','Gumbel(1.5)','FGM(0.7)']
AXES=[.15,.325,.5,.675,.85]
HS=[round(.15+.05*i,2) for i in range(8)]
SEED=20261023

def simulate(families=FAMILIES,reps=30,n=1200,axes=AXES,bandwidths=HS,
             theta_bound=15,quad_nodes=12,search_grid=17):
    rows=[]
    for fam in families:
        fami=FAMILIES.index(fam) # stable seed independent of subset
        for rep in range(reps):
            rng=np.random.default_rng(np.random.SeedSequence([SEED,fami,rep]))
            data=fam_sample(fam,n,rng)
            for u in axes:
                for v in axes:
                    truth=float(fam_target(fam,(u,v)))
                    for h in bandwidths:
                        frank=local_frank_fit(data,(u,v),h,kernel='rectangle',
                          theta_bound=theta_bound,quad_nodes=quad_nodes,search_grid=search_grid)
                        llb=jk_moment_rld(data,(u,v),h)
                        rows.append((fam,rep,u,v,h,truth,frank.estimated_r,llb,frank.hit_bound,frank.n_inside))
        print(f'Finished {fam}: {reps} replicates',flush=True)
    return pd.DataFrame(rows,columns=['family','rep','u','v','h','truth','frank','moment','frank_hit','ninside'])

def summarise(raw):
    overall=[]; spatial=[]
    for (family,h),sub in raw.groupby(['family','h'],sort=True):
        for field,name in [('frank','local Frank'),('moment','local log-bilinear')]:
            err=sub[field]-sub.truth
            overall.append(dict(family=family,h=h,method=name,ground_truth_mean=sub.truth.mean(),
                                estimate_mean=sub[field].mean(),bias=err.mean(),
                                rmse=np.sqrt(np.mean(err**2)),mae=np.mean(abs(err)),n=len(sub)))
    for (family,h,u,v),sub in raw.groupby(['family','h','u','v'],sort=True):
        for field,name in [('frank','local Frank'),('moment','local log-bilinear')]:
            err=sub[field]-sub.truth
            spatial.append(dict(family=family,h=h,u=u,v=v,method=name,
                                ground_truth=sub.truth.iloc[0],estimate_mean=sub[field].mean(),
                                bias=err.mean(),rmse=np.sqrt(np.mean(err**2)),n=len(sub)))
    return pd.DataFrame(overall),pd.DataFrame(spatial)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='results')
    ap.add_argument('--n',type=int,default=1200)
    ap.add_argument('--reps',type=int,default=30)
    ap.add_argument('--families',nargs='+',choices=FAMILIES,default=FAMILIES)
    ap.add_argument('--bandwidths',nargs='+',type=float,default=HS)
    ap.add_argument('--axes',nargs='+',type=float,default=AXES)
    ap.add_argument('--quad-nodes',type=int,default=12)
    ap.add_argument('--search-grid',type=int,default=17)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw=simulate(families=args.families,reps=args.reps,n=args.n,axes=args.axes,
                 bandwidths=args.bandwidths,quad_nodes=args.quad_nodes,search_grid=args.search_grid)
    overall,spatial=summarise(raw)
    raw.to_csv(out/'raw.csv',index=False)
    overall.to_csv(out/'summary.csv',index=False)
    spatial.to_csv(out/'spatial.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,fam in zip(axs.flat,args.families):
        for method,sty in [('local Frank','o-'),('local log-bilinear','s--')]:
            ss=overall[(overall.family==fam)&(overall.method==method)].sort_values('h')
            ax.plot(ss.h,ss.rmse,sty,label=method)
        ax.set(title=fam,xlabel='Bandwidth h',ylabel='Pooled RMSE')
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    for ax in axs.flat[len(args.families):]:ax.axis('off')
    fig.savefig(out/'rmse.png',dpi=150)
    plt.close(fig)
    print(overall.to_string(index=False),flush=True)
    print('Outputs:',out.resolve(),flush=True)

if __name__=='__main__':main()
