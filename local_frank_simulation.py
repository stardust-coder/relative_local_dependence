import numpy as np
from scipy.special import ndtr, ndtri
from relative_local_dependence.local_frank_fit import local_frank_fit, sample_frank
import pandas as pd

def sample_family(family,n,rng):
 if family=='Frank(3)': return sample_frank(n,3,int(rng.integers(1,2**31)))
 if family=='Clayton(1)':
  t=rng.gamma(1.,1.,size=n); e=rng.exponential(size=(n,2));return (1+e/t[:,None])**(-1.)
 if family=='Gaussian(0.3)':
  x=rng.normal(size=n); y=.3*x+np.sqrt(1-.3**2)*rng.normal(size=n);return np.stack([ndtr(x),ndtr(y)],axis=1)
 if family=='FGM(0.7)':
  u=rng.uniform(size=n); p=rng.uniform(size=n); a=.7*(1-2*u)
  # conditional CDF(v|u) = v+a*v*(1-v). Solve with stable quadratic
  v=2*p/(1+a+np.sqrt((1+a)**2-4*a*p))
  return np.stack([u,v],axis=1)
 raise ValueError(family)

def target(f,z):
 u,v=z
 if f=='Frank(3)':return 6.
 if f=='Clayton(1)':return 1.5*(1/u+1/v-1)
 if f=='FGM(0.7)':return 2.8/(1+.7*(1-2*u)*(1-2*v))**3
 if f=='Gaussian(0.3)':
  x,y=ndtri([u,v]); rho=.3
  phi=lambda t: np.exp(-t*t/2)/np.sqrt(2*np.pi)
  c=np.exp((2*rho*x*y-rho*rho*(x*x+y*y))/(2*(1-rho*rho)))/np.sqrt(1-rho*rho)
  return rho/((1-rho*rho)*phi(x)*phi(y)*c)

if __name__=='__main__':
 rng=np.random.default_rng(20261022)
 rows=[]; n=1200; reps=18
 for family in ['Frank(3)','Clayton(1)','Gaussian(0.3)','FGM(0.7)']:
  for k in range(reps):
   data=sample_family(family,n,rng)
   for z in [(0.35,0.6),(0.25,0.7)]:
    for h in [.2,.3]:
     for kernel in ['rectangle','biweight']:
      fit=local_frank_fit(data,z,h,kernel,theta_bound=15,quad_nodes=16,search_grid=25)
      rows.append(dict(family=family,rep=k,u=z[0],v=z[1],h=h,kernel=kernel,rtrue=target(family,z),rhat=fit.estimated_r,neff=fit.n_effective,hit=fit.hit_bound))
  print('completed',family,flush=True)
 df=pd.DataFrame(rows); df.to_csv('/mnt/data/local_frank_raw.csv',index=False)
 grp=df.groupby(['family','u','v','h','kernel']).agg(n=('rhat','size'),mean=('rhat','mean'),sd=('rhat','std'),rtrue=('rtrue','first'),neff=('neff','mean'),hit=('hit','mean')).reset_index()
 grouped=df.groupby(['family','u','v','h','kernel'])
 grp['rmse']=grouped.apply(lambda d:np.sqrt(np.mean((d.rhat-d.rtrue)**2)),include_groups=False).to_numpy()
 grp['wrong_sign']=grouped.apply(lambda d:np.mean(np.sign(d.rhat)!=np.sign(d.rtrue)),include_groups=False).to_numpy()
 grp['bias']=grp['mean']-grp['rtrue']
 grp.round(3).to_csv('/mnt/data/local_frank_summary.csv',index=False)
 print(grp.round(3).to_string(index=False))
