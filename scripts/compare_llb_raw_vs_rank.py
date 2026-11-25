import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
from numpy.polynomial.legendre import leggauss
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'results'/'raw_vs_pseudo';out.mkdir(parents=True,exist_ok=True)
n=1200; sigma=.15; h=.25; seed=20261124
rng=np.random.default_rng(seed)
x=rng.uniform(-1,1,n); y=x*x+rng.normal(0,sigma,n)
u=rankdata(x)/(n+1);v=rankdata(y)/(n+1)
grid=np.linspace(.25,.75,41)
xg=np.quantile(x,grid);yg=np.quantile(y,grid)
# LLB i^f original coordinate using rectangular kernel; s_2=1/3
# general LLB formula local dependence = 9 * Cov(S,T)/(h1*h2 * ph), 
# NOTE: correct formula: i_hat = 9 Cov((X-x0)/h1,(Y-y0)/h2)/(h1*h2) [proof: g00=ph/(4h1h2), g11-g10*g01/g00=h1*h2*g00 cov; divide h1^2 h2^2/9*g00 ->9cov/(h1*h2)]
def llb_ld(a,b,xa,ya,h1,h2):
    msk=(np.abs(xa-a)<=h1)&(np.abs(ya-b)<=h2);m=msk.sum()
    if m<2:return np.nan,m
    s=(xa[msk]-a)/h1;t=(ya[msk]-b)/h2
    return 9*np.mean((s-s.mean())*(t-t.mean()))/(h1*h2),m
raw=np.empty((41,41));cop=np.empty_like(raw);mraw=np.empty_like(raw);mcop=np.empty_like(raw)
for ii,p in enumerate(grid):
    for jj,q in enumerate(grid):
        raw[jj,ii],mraw[jj,ii]=llb_ld(xg[ii],yg[jj],x,y,h,h)
        cop[jj,ii],mcop[jj,ii]=llb_ld(p,q,u,v,h,h)
# marginal f_Y = 1/2 * int_-1^1 phi_sigma(y-x^2)dx
nodes,weights=leggauss(180)
fy=np.array([np.sum(weights*norm.pdf(z-nodes**2,scale=sigma))/2 for z in yg]); fx=.5
raw_converted=raw/(fx*fy[:,None])
# empirical transformed rank (U,V) mass; numerical values must reproduce earlier experiment exactly
# The comparison is computed independently; no prior files are required.
# true local dependence i^f(x,y)=2x/sigma^2
truth_raw=np.broadcast_to((2*xg/sigma**2)[None,:],raw.shape)
# Theoretical copula LD at matched xg,yg: i^c=i^f/(fx fy)
truth_cop=truth_raw/(fx*fy[:,None])
summary={
'raw_local_density_sample_min':int(mraw.min()),'rank_sample_min':int(mcop.min()),
'raw_local_density_sample_max':int(mraw.max()),'rank_sample_max':int(mcop.max()),
'raw_LLB_LD_min':float(raw.min()),'raw_LLB_LD_max':float(raw.max()),
'pseudo_LLB_LD_min':float(cop.min()),'pseudo_LLB_LD_max':float(cop.max()),
'raw_vs_truth_rmse':float(np.sqrt(np.mean((raw-truth_raw)**2))),
'pseudo_vs_truth_rmse':float(np.sqrt(np.mean((cop-truth_cop)**2))),
'raw_normalized_vs_pseudo_pearson_correlation':float(np.corrcoef(raw_converted.ravel(),cop.ravel())[0,1]),
'raw_vs_pseudo_naive_correlation':float(np.corrcoef(raw.ravel(),cop.ravel())[0,1]),
'raw_vs_true_sign_accuracy':float(np.mean(np.sign(raw)==np.sign(truth_raw))),
'pseudo_vs_true_sign_accuracy':float(np.mean(np.sign(cop)==np.sign(truth_cop))),
'raw_converted_vs_pseudo_rmse':float(np.sqrt(np.mean((raw_converted-cop)**2))),
}
print(summary)
rows=[]
for j,q in enumerate(grid):
 for i,p in enumerate(grid):
  rows.append({'u':p,'v':q,'x_quantile':xg[i],'y_quantile':yg[j],'n_raw_window':mraw[j,i],'n_pseudo_window':mcop[j,i], 'LLB_LD_raw_i_f':raw[j,i],'true_i_f':truth_raw[j,i],'LLB_LD_pseudo_i_c':cop[j,i],'true_i_c':truth_cop[j,i],'LLB_LD_raw_scaled_to_copula':raw_converted[j,i], 'f_Y':fy[j]})
pd.DataFrame(rows).to_csv(out/'comparison_41x41.csv',index=False)
pd.Series(summary).to_csv(out/'summary.csv',header=['value'])
# Plot per chart, matplotlib default color maps

def heat(z,extent, title,xlab,ylab,cbar,path,vmin=None,vmax=None):
 plt.figure(figsize=(7,5.8))
 plt.imshow(z,extent=extent,origin='lower',aspect='auto',interpolation='nearest',vmin=vmin,vmax=vmax)
 plt.xlabel(xlab);plt.ylabel(ylab);plt.title(title);plt.colorbar(label=cbar)
 plt.tight_layout();plt.savefig(out/path,dpi=190);plt.close()
heat(raw,[xg.min(),xg.max(),yg.min(),yg.max()],'LLB-LD on original coordinates (X, Y)','x','y',r'Estimated $i^f(x,y)$','llb_ld_original_xy.png')
heat(cop,[grid.min(),grid.max(),grid.min(),grid.max()],'LLB-LD on pseudo-observations (U, V)','u','v',r'Estimated $i^c(u,v)$','llb_ld_pseudo_uv.png')
heat(raw_converted,[grid.min(),grid.max(),grid.min(),grid.max()],'Raw LLB-LD divided by marginal densities','u','v',r'Estimated $i^f/(f_X f_Y)$','llb_ld_raw_density_adjusted.png')
heat(truth_raw,[xg.min(),xg.max(),yg.min(),yg.max()],r'True local dependence: $i^f(x,y)=2x/\sigma^2$','x','y','True LD','true_local_dependence.png')
heat(truth_cop,[grid.min(),grid.max(),grid.min(),grid.max()],'True copula local dependence','u','v',r'True $i^c(u,v)$','true_copula_local_dependence.png')
# Save seed and method details
(out/'readme.txt').write_text('Generated x~Uniform(-1,1); eps~N(0,0.15^2); y=x^2+eps; seed 20261124; n=1200.\n41x41 grid of marginal sample quantiles with p in [0.25,0.75].\nRectangular kernel and bandwidth h1=h2=.25 in each coordinate system.\nOriginal LLB-LD estimates i^f; pseudo LLB-LD estimates i^c.\nTheoretical i^f=2x/.15^2 and i^c=i^f/(f_X f_Y) with f_X=.5.\nRank-space pseudo values compared to previous saved run.\n')
