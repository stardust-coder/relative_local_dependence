import numpy as np
from local_dependence import sample_frank_normal_marginals
from relative_local_dependence import *
from scipy.stats import norm, rankdata


samples = sample_frank_normal_marginals(theta=3, n=50000, rng=np.random.default_rng(0))

pobs = np.column_stack([
    rankdata(samples[:, j]) / (samples.shape[0] + 1)
    for j in range(2)
])

band1 = get_bandwidth(pobs, band=None, kernel="gaussian")
band2 = get_bandwidth(samples, band=None, kernel="gaussian")

est1 = relative_local_dependence_frank_estimator(
    pobs,
    norm.cdf(0.0),
    norm.cdf(0.0),
    kernel_name="gaussian",
    band=band1,
    method="refined",
)

est2 = kernel_density_estimator(
    samples,
    0.0,
    0.0,
    kernel_name="gaussian",
    band=band2,
)

print("theta_hat:", est1 / 2)
print("est1:", est1)
print("est2:", est2)
print("product:", est1 * est2)