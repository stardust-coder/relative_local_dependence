"""Local Frank likelihood estimator with explicit boundary integration.

Dependencies: numpy, scipy. Run `python local_frank_fit.py` for a smoke test.
This estimates the *working-model* local Frank parameter; it is not in
 general a consistent estimator of true relative local dependence.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss


def frank_logpdf(u, v, theta):
    """Log Frank copula density. Supports positive and negative theta.

    Uses reflection c_{-t}(u,v)=c_t(u,1-v) to avoid negative-theta
    cancellation, and a nonnegative decomposition of the denominator.
    """
    u, v = np.broadcast_arrays(np.asarray(u, dtype=float), np.asarray(v, dtype=float))
    theta = float(theta)
    if abs(theta) < 1e-8:
        return np.zeros_like(u)
    t = abs(theta)
    if theta < 0:
        v = 1.0 - v
    # q = e^{-tu}(1-e^{-tv}) + e^{-tv}(1-e^{-t(1-v)})
    with np.errstate(divide='ignore', invalid='ignore'):
        log_a = -t*u + np.log(-np.expm1(-t*v))
        log_b = -t*v + np.log(-np.expm1(-t*(1.0-v)))
        log_q = np.logaddexp(log_a, log_b)
        out = np.log(t) + np.log(-np.expm1(-t)) - t*(u+v) - 2*log_q
    return out


def kernel_1d(x, kind):
    if kind == 'rectangle':
        return 0.5 * (np.abs(x) <= 1.0)
    if kind == 'biweight':
        return (15.0/16.0) * np.maximum(1.0-x*x, 0.0)**2 * (np.abs(x) <= 1.0)
    raise ValueError('kernel must be rectangle or biweight')


@dataclass
class LocalFrankResult:
    theta: float
    estimated_r: float
    objective: float
    n_inside: int
    n_effective: float
    hit_bound: bool
    kernel: str
    bandwidth: float


def _quadrature(center, h, nodes=32, kernel='biweight'):
    # Coordinates are in kernel units; clipping gives exact truncation at 0,1.
    x0, y0 = map(float, center)
    edges = []
    for a in (x0, y0):
        left, right = max(-1.0, -a/h), min(1.0, (1.0-a)/h)
        if right <= left:
            raise ValueError('target location must lie in the unit square')
        edges.append((left, right))
    nodes1, weights1 = leggauss(nodes)
    mapped = []
    for left, right in edges:
        t = 0.5*(right-left)*nodes1 + 0.5*(left+right)
        w = 0.5*(right-left)*weights1*kernel_1d(t, kernel)
        mapped.append((t,w))
    (tx,wx),(ty,wy)=mapped
    xx, yy=np.meshgrid(x0+h*tx, y0+h*ty, indexing='ij')
    ww=np.outer(wx,wy)
    return xx.ravel(), yy.ravel(), ww.ravel()


def local_frank_fit(data, at, bandwidth, kernel='biweight', theta_bound=25,
                    quad_nodes=32, search_grid=81):
    """Optimize sample local likelihood by multistart bounded scalar search.

    Objective = sum K_h(z_i-at) log c_theta(z_i)/n
              - integral_[0,1]^2 K_h(w-at)c_theta(w) dw.
    Uses tensor Gauss-Legendre quadrature, clipped to the valid support.
    """
    data=np.asarray(data,dtype=float)
    at=np.asarray(at,dtype=float)
    if data.ndim!=2 or data.shape[1]!=2 or not np.all((0 <= data)&(data <= 1)):
        raise ValueError('data must be an n-by-2 matrix in [0,1]^2')
    if at.shape != (2,) or not np.all((0 < at)&(at < 1)):
        raise ValueError('at must be a point in (0,1)^2')
    if bandwidth<=0 or theta_bound<=0:
        raise ValueError('bandwidth, theta_bound must be positive')
    h=float(bandwidth)
    residual=(data-at)/h
    weights=kernel_1d(residual[:,0],kernel)*kernel_1d(residual[:,1],kernel)/(len(data)*h*h)
    keep=weights>0
    ww=weights[keep]
    pts=data[keep]
    qx,qy,qw=_quadrature(at,h,nodes=quad_nodes,kernel=kernel)
    def objective(theta):
        sample=np.sum(ww*frank_logpdf(pts[:,0],pts[:,1],theta))
        integral=np.sum(qw*np.exp(frank_logpdf(qx,qy,theta)))
        return float(sample-integral)
    grid=np.linspace(-theta_bound,theta_bound,search_grid)
    vals=np.array([objective(t) for t in grid])
    # Grid search is ONLY for locating peaks, not the final estimator.
    candidates=[0, search_grid-1]
    candidates.extend([i for i in range(1,search_grid-1) if vals[i]>=vals[i-1] and vals[i]>=vals[i+1]])
    best=(float(vals[np.argmax(vals)]),float(grid[np.argmax(vals)]))
    for i in candidates:
        lo=grid[max(0,i-1)]
        hi=grid[min(search_grid-1,i+1)]
        if lo==hi: continue
        result=minimize_scalar(lambda t:-objective(t),bounds=(lo,hi),method='bounded',
                               options={'xatol':1e-6})
        if -result.fun>best[0]:
            best=(-float(result.fun),float(result.x))
    n_eff=(np.sum(ww)**2 / np.sum(ww**2)) if len(ww) else 0.0
    return LocalFrankResult(theta=best[1], estimated_r=2*best[1], objective=best[0],
                            n_inside=int(np.sum(keep)), n_effective=float(n_eff),
                            hit_bound=abs(best[1]) >= 0.98*theta_bound,
                            kernel=kernel, bandwidth=h)


def sample_frank(n, theta, seed=42):
    rng=np.random.default_rng(seed)
    u=rng.uniform(size=n)
    p=rng.uniform(size=n)
    if abs(theta)<1e-8:
        return np.column_stack([u,p])
    a=np.expm1(-theta*u)
    d=np.expm1(-theta)
    b=p*d/(np.exp(-theta*u)-p*a)
    v=-np.log1p(b)/theta
    return np.column_stack([u,v])


if __name__=='__main__':
    # Sanity: integrate each density close to one over unit square.
    gx, gw=leggauss(100)
    gg=0.5*(gx+1)
    gridx,gridy=np.meshgrid(gg,gg,indexing='ij')
    W=np.outer(gw,gw)*0.25
    for t in [-8.,-2.,0.,2.,8.]:
        mass=np.sum(W*np.exp(frank_logpdf(gridx,gridy,t)))
        print(f'Frank(theta={t:>4g}) mass: {mass:.8f}')
        assert abs(mass-1)<1e-5
    # 2 valid test settings. No true parameter is passed to optimizer.
    data=sample_frank(1500,theta=3.0,seed=11)
    for k in ('rectangle','biweight'):
        print(k,local_frank_fit(data,at=(.35,.6),bandwidth=.22,kernel=k))
