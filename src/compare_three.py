import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtri, ndtr
from scipy.stats import kendalltau
from numpy.polynomial.legendre import leggauss
from concurrent.futures import ProcessPoolExecutor, as_completed

FAMS=['Frank(3)','Clayton(1)','Gaussian(0.3)','Gumbel(1.5)','FGM(0.7)']
GRID=np.array([.1,.3,.5,.7,.9])
CENTERS=np.array([(a,b) for a in GRID for b in GRID])
HS=np.round(np.arange(.15,.501,.05),2)
QN=8
nodes,weights=leggauss(QN)

# Exact copula simulation

def sample(fi,n,rng):
    U=rng.uniform(size=n); W=rng.uniform(size=n)
    if fi==0:
        theta=3.; a=np.exp(-theta*U); B=W*np.expm1(-theta)/(a-W*(a-1))
        return np.column_stack((U,-np.log1p(B)/theta))
    if fi==1:
        fr=rng.gamma(1.,1.,size=n)
        return (1+rng.exponential(size=(n,2))/fr[:,None])**(-1.)
    if fi==2:
        x=rng.standard_normal(n); y=.3*x+np.sqrt(1-.3**2)*rng.standard_normal(n)
        return np.column_stack((ndtr(x),ndtr(y)))
    if fi==3:
        aa=1/1.5; th=rng.uniform(1e-10,np.pi-1e-10,size=n); e=rng.exponential(size=n)
        stable=(np.sin(aa*th)/np.sin(th)**(1/aa))*(np.sin((1-aa)*th)/e)**((1-aa)/aa)
        z=np.exp(-(rng.exponential(size=(n,2))/stable[:,None])**aa)
        return np.clip(z,1e-14,1-1e-14)
    A=.7*(1-2*U)
    vv=2*W/(1+A+np.sqrt((1+A)**2-4*A*W))
    return np.column_stack((U,vv))

def log_density(fi,u,v):
    u=np.asarray(u);v=np.asarray(v)
    if fi==0:
        th=3.; a=np.exp(-th*u); b=np.exp(-th*v); z=np.expm1(-th)+(a-1)*(b-1)
        return np.log(th*(-np.expm1(-th))) -th*(u+v)-2*np.log(-z)
    if fi==1:
        z=1/u+1/v-1
        return np.log(2)-2*np.log(u)-2*np.log(v)-3*np.log(z)
    if fi==2:
        x=ndtri(u);y=ndtri(v);rho=.3
        return -.5*np.log1p(-rho*rho)+(2*rho*x*y-rho*rho*(x*x+y*y))/(2*(1-rho*rho))
    if fi==3:
        th=1.5;a=-np.log(u);b=-np.log(v); s=a**th+b**th;t=s**(1/th)
        return -t+(th-1)*(np.log(a)+np.log(b))-np.log(u)-np.log(v)+(2/th-2)*np.log(s)+np.log1p((th-1)/t)
    return np.log1p(.7*(1-2*u)*(1-2*v))

def truth(fi,u,v):
    if fi==0:return 6.
    if fi==1:return 1.5*(1/u+1/v-1)
    if fi==2:
        x=ndtri(u);y=ndtri(v)
        return .3/(1-.3**2)/ (np.exp(-.5*x*x)/np.sqrt(2*np.pi)*np.exp(-.5*y*y)/np.sqrt(2*np.pi)*np.exp(log_density(fi,u,v)))
    if fi==4:
        c=np.exp(log_density(fi,u,v));return 2.8/c**3
    # mixed derivative of log copula via fourth-order central finite difference
    eps=2e-4
    f=lambda a,b: log_density(fi,a,b)
    D=lambda hu,hv: (f(u+hu,v+hv)-f(u+hu,v-hv)-f(u-hu,v+hv)+f(u-hu,v-hv))/(4*hu*hv)
    i=(4*D(eps,eps)-D(2*eps,2*eps))/3
    return i/np.exp(f(u,v))

def eandder(b,x):
    if abs(b)<1e-4:
        return x+b*x*x/2+b*b*x**3/6+b**3*x**4/24, x*x/2+b*x**3/3+b*b*x**4/8+b**3*x**5/30
    z=b*x; ex=np.exp(z)
    return np.expm1(z)/b,(b*x*ex-np.expm1(z))/b**2

def calc(v,xy,needgrad=True):
    b,c,d=v; x=xy[:,0];y=xy[:,1]
    eb,db=eandder(b,x);ec,dc=eandder(c,y)
    denom=1-.5*d*eb*ec
    if np.any(denom<=.015) or not np.all(np.isfinite(denom)):return None
    lq=b*x+c*y-2*np.log(denom)
    if np.max(lq)>30:return None
    if not needgrad:return lq
    scores=np.column_stack((x+d*db*ec/denom,y+d*eb*dc/denom,eb*ec/denom))
    return lq,scores

def fit_lf(X,n,h,bounds_rect,starts=1):
    m=len(X)
    if m<7:return np.nan,False,False
    left,right,bottom,top=bounds_rect
    qx=(right-left)*nodes/2+(right+left)/2
    qy=(top-bottom)*nodes/2+(top+bottom)/2
    gx,gy=np.meshgrid(qx,qy,indexing='ij')
    Q=np.stack((gx.ravel(),gy.ravel()),axis=1)
    weights2=np.outer(weights,weights).ravel()*(right-left)*(top-bottom)/4
    mean=X.mean(axis=0);cov=np.mean((X[:,0]-mean[0])*(X[:,1]-mean[1]))
    varx=np.var(X[:,0]);vary=np.var(X[:,1]);d0=np.clip(cov/max(varx*vary,1e-6),-1.,1.)
    bx=np.clip(mean[0]*3,-1.3,1.3);by=np.clip(mean[1]*3,-1.3,1.3)
    bounds=[(-1.5,1.5),(-1.5,1.5),(-1.45,1.45)]
    # Coarse grid screening, independent of the true copula parameters.
    def fun(v,grad):
        cq=calc(v,Q,True)
        if cq is None:return None
        cx=calc(v,X,True)
        if cx is None:return None
        lq,sq=cq; lx,sx=cx
        masses=weights2*np.exp(lq);Z=np.sum(masses)
        if Z<=0:return None
        value=m*np.log(Z)-np.sum(lx)
        gradient=m*(masses@sq)/Z-sx.sum(axis=0)
        return value,gradient
    candidates=np.array([(b,c,d) for b in [-1,0,1] for c in [-1,0,1] for d in [-1,0,1]],dtype=float)
    candidates=np.vstack([candidates,np.array([bx,by,d0]),np.array([0,0,0])])
    values=[fun(v,False) for v in candidates]
    ranked=sorted([(res[0],i) for i,res in enumerate(values) if res is not None])
    if not ranked:return np.nan,False,False
    best=None
    for _,k in ranked[:starts]:
        init=candidates[k]
        def f(v):
            z=fun(v,True)
            if z is None:return 1e6+np.sum(v*v)*1e3,2e3*v
            return z
        result=minimize(f,init,method='L-BFGS-B',jac=True,bounds=bounds,options={'maxiter':120,'ftol':1e-10,'gtol':1e-6})
        z=fun(result.x,True)
        if z is None:continue
        if best is None or z[0]<best[0]:best=(z[0],result,z[1])
    if best is None:return np.nan,False,False
    _,res,gr=best
    v=res.x; b,c,d=v
    Z=np.sum(weights2*np.exp(calc(v,Q,False)))
    # eliminate alpha analytically: exp(alpha)=m/(n*h^2* integral_Q exp(log_q))
    ea=m/(n*h*h*Z)
    r=d/(h*h*ea)
    return r,bool(res.success or np.linalg.norm(gr)<5e-4),bool(np.any(np.abs(v)>[1.48,1.48,1.43]))

def simulate_job(job):
    fi,rep,kind,starts=job;n=1200
    rng=np.random.default_rng(np.random.SeedSequence([20261120,fi,rep]))
    D=sample(fi,n,rng)
    if kind=='ranks':
        from scipy.stats import rankdata
        D=np.column_stack((rankdata(D[:,0])/(n+1),rankdata(D[:,1])/(n+1)))
    result=[]
    for h in HS:
        for u,v in CENTERS:
            mask=(np.abs(D[:,0]-u)<=h)&(np.abs(D[:,1]-v)<=h)
            X=(D[mask]-[u,v])/h;m=len(X);ph=m/n
            if m<7:continue
            cov=np.mean((X[:,0]-X[:,0].mean())*(X[:,1]-X[:,1].mean()))
            llb=36*cov/ph
            tau=kendalltau(X[:,0],X[:,1]).statistic
            lkt=18*tau/ph if np.isfinite(tau) else np.nan
            # Clip the integration domain to copula support. Note that this differs
            # from the theoretical symmetric-square local model at boundary points.
            rect=(max(-1,-u/h), min(1,(1-u)/h),max(-1,-v/h), min(1,(1-v)/h))
            lf,success,near=fit_lf(X,n,h,rect,starts)
            interior=(u>=h-1e-12 and 1-u>=h-1e-12 and v>=h-1e-12 and 1-v>=h-1e-12)
            result.append((FAMS[fi],rep,kind,h,u,v,interior,m,truth(fi,u,v),lkt,llb,lf,success,near))
    return result

def summarize(df):
    rows=[]
    for (fam,kind,h,interior),g in df.groupby(['family','input','h','interior']):
        for meth in ['LKT','LLB','LF']:
            w=g[np.isfinite(g[meth])]
            if not len(w):continue
            e=w[meth]-w.truth
            rows.append(dict(family=fam,input=kind,h=h,interior=interior,method=meth,nvals=len(w),ncenters=w[['u','v']].drop_duplicates().shape[0],RMSE=np.sqrt(np.mean(e**2)),Bias=np.mean(e),MAE=np.mean(np.abs(e)),Success=(w.LF_success.mean() if meth=='LF' else np.nan),Bound=(w.LF_bound.mean() if meth=='LF' else np.nan)))
    return pd.DataFrame(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reps',type=int,default=30)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--input',choices=['true','ranks'],default='true')
    parser.add_argument('--starts',type=int,default=1)
    parser.add_argument('--out',default='results/copula_grid')
    parser.add_argument('--families',default='0,1,2,3,4')
    args=parser.parse_args(); os.makedirs(args.out,exist_ok=True)
    jobs=[(int(fi),rep,args.input,args.starts) for fi in args.families.split(',') for rep in range(args.reps)]
    start=time.time(); data=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fs=[ex.submit(simulate_job,j) for j in jobs]
        for i,f in enumerate(as_completed(fs),1):
            data.extend(f.result())
            if i%10==0:
                print('done',i,'of',len(jobs),'elapsed',round(time.time()-start,1),flush=True)
                pd.DataFrame(data,columns=['family','rep','input','h','u','v','interior','m','truth','LKT','LLB','LF','LF_success','LF_bound']).to_csv(args.out+'/partial.csv',index=False)
    df=pd.DataFrame(data,columns=['family','rep','input','h','u','v','interior','m','truth','LKT','LLB','LF','LF_success','LF_bound'])
    df.sort_values(['family','rep','h','u','v']).to_csv(args.out+'/raw.csv',index=False)
    summ=summarize(df)
    summ.to_csv(args.out+'/summary_by_region.csv',index=False)
    allsum=[]
    for (fam,kind,h),g in df.groupby(['family','input','h']):
        for meth in ['LKT','LLB','LF']:
            w=g[np.isfinite(g[meth])];e=w[meth]-w.truth
            allsum.append(dict(family=fam,input=kind,h=h,method=meth,nvals=len(w),RMSE=np.sqrt(np.mean(e**2)),Bias=e.mean(),MAE=np.mean(abs(e)),Success=(w.LF_success.mean() if meth=='LF' else np.nan)))
    pd.DataFrame(allsum).to_csv(args.out+'/summary_all.csv',index=False)
    print('FINISHED',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
