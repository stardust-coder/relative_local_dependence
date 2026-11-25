import numpy as np,pandas as pd, time
from scipy.special import ndtr,ndtri
from scipy.stats import levy_stable
from relative_local_dependence.local_frank_fit import local_frank_fit, sample_frank
from relative_local_dependence.local_frank_simulation import sample_family, target

GAMMA=1.5

def gumbel_sample(n,rng,gamma=GAMMA):
    alpha=1/gamma
    scale=np.cos(np.pi*alpha/2)**(1/alpha)
    S=levy_stable.rvs(alpha,1,scale=scale,size=n,random_state=rng)
    E=rng.exponential(size=(n,2))
    return np.exp(-(E/S[:,None])**alpha)

def gumbel_density_r(u,v,gamma=GAMMA):
    x,y=-np.log(u),-np.log(v)
    a=x**gamma+y**gamma; p=1/gamma; b=gamma-1
    t=a**p
    c=np.exp(-t)*(x*y)**(gamma-1)*a**(2*p-2)*(1+b/t)/(u*v)
    D=1+b*a**(-p)
    Fpp=-p*(p-1)*a**(p-2)-(2*p-2)/a**2+b*p*(p+1)*a**(-p-2)/D-b*b*p*p*a**(-2*p-2)/D**2
    i=(gamma**2*x**(gamma-1)*y**(gamma-1)*Fpp)/(u*v)
    return i/c

def fam_sample(f,n,rng):
    if f=='Gumbel(1.5)': return gumbel_sample(n,rng)
    return sample_family(f,n,rng)

def fam_target(f,z):
    if f=='Gumbel(1.5)':return gumbel_density_r(*z)
    return target(f,z)

if __name__=='__main__':
    rng=np.random.default_rng(20261022)
    n=1200; reps=80; h=.25
    axes=[.15,.325,.5,.675,.85]
    families=['Frank(3)','Clayton(1)','Gaussian(0.3)','Gumbel(1.5)','FGM(0.7)']
    rows=[]; start=time.time()
    for family in families:
        for rep in range(reps):
            sample=fam_sample(family,n,rng)
            for u in axes:
                for v in axes:
                    z=(u,v)
                    fit=local_frank_fit(sample,z,h,'rectangle',theta_bound=15,quad_nodes=20,search_grid=25)
                    rows.append({'family':family,'rep':rep,'u':u,'v':v,'h':h,'n':n,'rtrue':fam_target(family,z),'rhat':fit.estimated_r,'ninside':fit.n_inside,'hit':fit.hit_bound})
        print(family,'done',round(time.time()-start,1),'sec',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv('/mnt/data/local_frank_spatial_raw.csv',index=False)
    g=df.groupby(['family','u','v','h','n']).agg(reps=('rhat','size'),mean=('rhat','mean'),sd=('rhat','std'),rtrue=('rtrue','first'),mean_inside=('ninside','mean'),bound_rate=('hit','mean')).reset_index()
    g['bias']=g['mean']-g['rtrue']
    g['rmse']=np.sqrt(g['bias']**2+(g['reps']-1)/g['reps']*g['sd']**2)
    g['wrong_sign']=df.groupby(['family','u','v','h','n']).apply(lambda x: np.mean(np.sign(x.rhat)!=np.sign(x.rtrue)),include_groups=False).to_numpy()
    g.to_csv('/mnt/data/local_frank_spatial_summary.csv',index=False)
    print(g.groupby('family').agg(mean_abs_bias=('bias',lambda x:np.mean(abs(x))),mean_rmse=('rmse','mean'),max_rmse=('rmse','max'),boundary=('bound_rate','mean')).round(3),flush=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    for family in families:
        d=g[g.family==family]
        for metric in ['bias','rmse']:
            grid=d.pivot(index='v',columns='u',values=metric).sort_index(ascending=False)
            fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
            vals=grid.values
            if metric=='bias':
                bound=max(np.max(abs(vals)),.01)
                im=ax.imshow(vals,cmap='RdBu_r',vmin=-bound,vmax=bound,origin='upper')
            else: im=ax.imshow(vals,cmap='viridis',origin='upper',vmin=0)
            ax.set_xticks(range(len(axes)),[str(x) for x in axes]);ax.set_yticks(range(len(axes)),[str(x) for x in axes[::-1]])
            ax.set_xlabel('u');ax.set_ylabel('v');ax.set_title(f'{family}: {metric} (n={n}, h={h}, reps={reps})')
            for i in range(len(axes)):
                for j in range(len(axes)):
                    ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if metric=='bias' and abs(vals[i,j])>bound*.65 else 'black')
            fig.colorbar(im,ax=ax,label=metric)
            fig.savefig('/mnt/data/local_frank_'+family.split('(')[0].lower()+'_'+metric+'.png',dpi=145)
            plt.close(fig)
    print('complete',round(time.time()-start,1),flush=True)
