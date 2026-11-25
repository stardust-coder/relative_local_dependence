"""Regression checks for historical published tables (three-decimal precision)."""
from pathlib import Path
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]
t=root/'tables'
checks={
 'bias_variance_rmse_h025_interior.csv':{
   ('Frank(3)','RMSE'):(2.964,2.842,2.836),
   ('Clayton(1)','RMSE'):(2.882,2.614,2.570),
   ('Gaussian(0.3)','RMSE'):(3.043,2.967,2.931),
   ('Gumbel(1.5)','RMSE'):(2.690,2.495,2.481),
   ('FGM(0.7)','RMSE'):(2.879,2.873,2.869)},
 'rmse_all25_every_bandwidth.csv':{
   ('Frank(3)',.30):(5.338,3.395,5.018),
   ('Clayton(1)',.30):(6.732,7.785,7.344),
   ('Gaussian(0.3)',.30):(5.646,5.957,5.343),
   ('Gumbel(1.5)',.30):(5.726,5.545,6.609),
   ('FGM(0.7)',.30):(4.390,4.422,4.352)}
}
for fn,cases in checks.items():
 d=pd.read_csv(t/fn)
 for (fam,tag),vals in cases.items():
  mask=(d.family==fam)&((d.metric==tag) if isinstance(tag,str) else np.isclose(d.h,tag))
  assert mask.sum()==1,(fn,fam,tag)
  actual=d.loc[mask,['LKT','LLB','LF']].iloc[0].to_numpy()
  assert np.allclose(actual,vals,atol=.000501,rtol=0),(fn,fam,actual,vals)
assert len(pd.read_csv(t/'rmse_all25_every_bandwidth.csv'))==40
assert len(pd.read_csv(t/'rmse_bandwidth_interior.csv'))==20
assert len(pd.read_csv(t/'rmse_sample_size.csv'))==15
print('PASS: all checks for paper tables and row counts')
