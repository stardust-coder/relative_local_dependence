import os
os.environ['OPENBLAS_NUM_THREADS']='1'; os.environ['OMP_NUM_THREADS']='1'
import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.compare_three import sample, truth, fit_lf
import numpy as np,pandas as pd
from scipy.stats import kendalltau
from concurrent.futures import ProcessPoolExecutor,as_completed
OUT='results/normality'
os.makedirs(OUT,exist_ok=True)
FAMS=['Frank(3)','Clayton(1)','Gaussian(0.3)']
# h(n)=0.25 (n/4800)^(-0.12); n h^6 increases
NVALS=[30000,100000]
B=150

def job(k):
 fi,n,rep=k
 h=.25*(n/4800)**(-.12)
 rng=np.random.default_rng(np.random.SeedSequence([20261123,7777,fi,n,rep]))
 D=sample(fi,n,rng)
 x=(D-.5)/h
 X=x[(abs(x[:,0])<=1)&(abs(x[:,1])<=1)]
 m=len(X); ph=m/n
 cov=np.mean((X[:,0]-X[:,0].mean())*(X[:,1]-X[:,1].mean()))
 llb=36*cov/ph
 tau=kendalltau(X[:,0],X[:,1]).statistic
 lkt=18*tau/ph
 lf,conv,bound=fit_lf(X,n,h,(-1,1,-1,1))
 return dict(family=FAMS[fi],n=n,rep=rep,h=h,u=.5,v=.5,m=m,truth=truth(fi,.5,.5),LKT=lkt,LLB=llb,LF=lf,LF_success=conv,LF_bound=bound)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--ns',default='30000,100000');p.add_argument('--reps',type=int,default=150);p.add_argument('--workers',type=int,default=6);p.add_argument('--out',default='results/normality');args=p.parse_args()
 NVALS=[int(a) for a in args.ns.split(',')];B=args.reps;OUT=args.out;os.makedirs(OUT,exist_ok=True)
 jobs=[(fi,n,rep) for fi in range(3) for n in NVALS for rep in range(B)]
 start=time.time(); rows=[]
 with ProcessPoolExecutor(max_workers=args.workers) as ex:
  fs=[ex.submit(job,x) for x in jobs]
  for j,f in enumerate(as_completed(fs),1):
   rows.append(f.result())
   if j%50==0:
    print('completed',j,'/',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
    pd.DataFrame(rows).to_csv(OUT+'/partial.csv',index=False)
 pd.DataFrame(rows).sort_values(['family','n','rep']).to_csv(OUT+'/raw_new.csv',index=False)
 print('finished',round(time.time()-start,1),flush=True)
