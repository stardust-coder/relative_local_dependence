"""Paired nonparametric bootstrap dependence maps for Y=X^2+epsilon.
Rebuild pseudo-observations on each bootstrap sample, preserving rank uncertainty.
LLB-LD original i^f; LLB-LD copula i^c; LF copula RLD r^c.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.compare_three import fit_lf
import numpy as np,pandas as pd
from scipy.stats import rankdata
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
OUT=Path('results/bootstrap_map');OUT.mkdir(parents=True,exist_ok=True)
N=1200; H=.25; G=17; B=61; SEED=20261124
rng=np.random.default_rng(SEED)
X=rng.uniform(-1,1,N); Y=X*X+rng.normal(0,.15,N)
GRID=np.linspace(.25,.75,G)
XG=np.quantile(X,GRID);YG=np.quantile(Y,GRID)
CENTERS=np.array([(i,j) for j in range(G) for i in range(G)])

def estimate(x,y):
    n=len(x)
    u=rankdata(x,method='average')/(n+1);v=rankdata(y,method='average')/(n+1)
    out=np.full((3,G*G),np.nan)
    uv=np.column_stack((u,v))
    for k,(i,j) in enumerate(CENTERS):
        # raw original coordinates, h=0.25 in each dimension
        ma=(np.abs(x-XG[i])<=H)&(np.abs(y-YG[j])<=H)
        if ma.sum()>=7:
            s=(x[ma]-XG[i])/H;t=(y[ma]-YG[j])/H
            out[0,k]=9*np.mean((s-s.mean())*(t-t.mean()))/(H*H)
        # copula coordinates
        ub,vb=GRID[i],GRID[j]
        mc=(np.abs(u-ub)<=H)&(np.abs(v-vb)<=H)
        if mc.sum()>=7:
            z=(uv[mc]-[ub,vb])/H
            out[1,k]=9*np.mean((z[:,0]-z[:,0].mean())*(z[:,1]-z[:,1].mean()))/(H*H)
            out[2,k]=fit_lf(z,n,H,(-1,1,-1,1),starts=1)[0]
    return out

def job(rep):
    rng=np.random.default_rng(np.random.SeedSequence([SEED,2026,rep]))
    ids=rng.integers(0,N,size=N)
    return rep,estimate(X[ids],Y[ids])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reps',type=int,default=61);p.add_argument('--workers',type=int,default=4);p.add_argument('--out',default='results/bootstrap_map');args=p.parse_args()
    B=args.reps;OUT=Path(args.out);OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    ref=estimate(X,Y)
    print('original done',round(time.time()-start,2),flush=True)
    boots=np.full((B,3,G*G),np.nan)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(job,i) for i in range(B)]
        for idx,f in enumerate(as_completed(fs),1):
            r,a=f.result();boots[r]=a
            if idx%20==0:
                print('completed',idx,'/',B,'time',round(time.time()-start,1),'s',flush=True)
                np.savez_compressed(OUT/'partial.npz',original=ref,bootstrap=boots,grid=GRID,xgrid=XG,ygrid=YG)
    np.savez_compressed(OUT/'bootstrap.npz',original=ref,bootstrap=boots,grid=GRID,xgrid=XG,ygrid=YG,x=X,y=Y)
    print('ALL DONE',round(time.time()-start,1),flush=True)
