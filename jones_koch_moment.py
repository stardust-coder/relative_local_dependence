"""Rectangle-kernel Jones--Koch-style moment estimator.

Matches the weighted-moment construction in stardust-coder's
local_dependence.py, with uniform kernel and user-specified bandwidth.
For RLD, divides its local dependence estimate by g00 (density estimate).
No marginal rank transformation or boundary correction is applied.
"""
from __future__ import annotations
import numpy as np


def jk_moment_rld(data, point, h, *, min_count=2, return_details=False):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] == 0:
        raise ValueError('data must be a nonempty n x 2 array')
    h1,h2 = (float(h),float(h)) if np.isscalar(h) else map(float,h)
    if h1 <= 0 or h2 <= 0:
        raise ValueError('bandwidths must be positive')
    x,y=map(float,point)
    dx=data[:,0]-x
    dy=data[:,1]-y
    sel=(np.abs(dx)<=h1)&(np.abs(dy)<=h2)
    count=int(sel.sum())
    if count < min_count:
        result=float('nan')
        return (result,dict(count=count,density=float('nan'))) if return_details else result
    # Product rectangular kernel: 1/(4 h1 h2) in local rectangle.
    w=1.0/(4*h1*h2)
    fhat=w*count/len(data)
    # Stable equivalent of (g11-g10*g01/g00)/g00; covariance of selected points.
    covariance=float(np.mean(dx[sel]*dy[sel])-np.mean(dx[sel])*np.mean(dy[sel]))
    mu2=1.0/3.0
    ihat=covariance/(h1*h1*h2*h2*mu2*mu2)
    rhat=ihat/fhat
    return (rhat,dict(count=count,density=fhat,local_dependence=ihat,local_covariance=covariance)) if return_details else rhat


def jk_moment_grid(data, grid, bandwidths):
    """Return estimates indexed by (h, u, v)."""
    return {(float(h),float(u),float(v)):jk_moment_rld(data,(u,v),h)
            for h in bandwidths for u,v in grid}
