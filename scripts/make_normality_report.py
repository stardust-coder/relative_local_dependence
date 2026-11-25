"""Generate exploratory normality metrics from archived Monte Carlo estimates."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import shapiro,jarque_bera,skew,kurtosis
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'tables';OUT.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/'data/normality_all_samples.csv')
rows=[]
for (fam,n),g in df.groupby(['family','n']):
    for meth in ['LKT','LLB','LF']:
        v=g[meth].dropna().to_numpy()
        sd=v.std(ddof=1)
        z=(v-v.mean())/sd
        rows.append(dict(family=fam,n=n,method=meth,B=len(v),h=g.h.iloc[0],mean=v.mean(),sd=sd,rmse=np.sqrt(np.mean((v-g.truth.iloc[0])**2)),skewness=skew(z,bias=False),excess_kurtosis=kurtosis(z,bias=False),shapiro_p=shapiro(z).pvalue,jarque_bera_p=jarque_bera(z).pvalue,scaled_sd=sd*np.sqrt(n*g.h.iloc[0]**6)))
pd.DataFrame(rows).to_csv(OUT/'normality_summary.csv',index=False)
print('Normality summary saved; Shapiro p-values:')
print(pd.DataFrame(rows).pivot(index=['family','n'],columns='method',values='shapiro_p').round(3))
