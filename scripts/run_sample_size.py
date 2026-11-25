import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time,argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.stats import kendalltau
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import compare_three as model
H=.25
CENTERS=[(u,v) for u in [.3,.5,.7] for v in [.3,.5,.7]]

def job(params):
    fi,n,rep=params
    rng=np.random.default_rng(np.random.SeedSequence([20261121,fi,n,rep]))
    D=model.sample(fi,n,rng)
    out=[]
    for u,v in CENTERS:
        X=(D[(np.abs(D[:,0]-u)<=H)&(np.abs(D[:,1]-v)<=H)]-[u,v])/H
        m=len(X);ph=m/n
        if m<7:continue
        cov=np.mean((X[:,0]-X[:,0].mean())*(X[:,1]-X[:,1].mean()))
        llb=36*cov/ph
        tau=kendalltau(X[:,0],X[:,1]).statistic
        lkt=18*tau/ph if np.isfinite(tau) else np.nan
        lf,ok,bound=model.fit_lf(X,n,H,(-1,1,-1,1),1)
        out.append((model.FAMS[fi],n,rep,H,u,v,m,model.truth(fi,u,v),lkt,llb,lf,ok,bound))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--ns',default='300,4800');p.add_argument('--reps',type=int,default=200);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',default='results/sample_size');args=p.parse_args()
    ns=[int(x) for x in args.ns.split(',')]
    jobs=[(fi,n,rep) for n in ns for fi in range(5) for rep in range(args.reps)]
    os.makedirs(args.out,exist_ok=True)
    start=time.time(); records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        future=[ex.submit(job,j) for j in jobs]
        for k,f in enumerate(as_completed(future),1):
            records.extend(f.result())
            if k%50==0:
                print('done',k,'/',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
                pd.DataFrame(records,columns=['family','n','rep','h','u','v','m','truth','LKT','LLB','LF','LF_success','LF_bound']).to_csv(args.out+'/partial.csv',index=False)
    df=pd.DataFrame(records,columns=['family','n','rep','h','u','v','m','truth','LKT','LLB','LF','LF_success','LF_bound']);df.to_csv(args.out+'/raw_new.csv',index=False)
    print('complete time',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
