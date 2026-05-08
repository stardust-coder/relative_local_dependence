import numpy as np
from math import sqrt, pi
from scipy import integrate, optimize


EPS = 1e-10


# ============================================================
# Frank copula
# ============================================================

def frank_cdf(theta, u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if abs(theta) < EPS:
        return u * v

    num = np.expm1(-theta * u) * np.expm1(-theta * v)
    den = np.expm1(-theta)
    return -np.log1p(num / den) / theta


def frank_density(theta, u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    if abs(theta) < EPS:
        return np.ones(np.broadcast(u, v).shape)

    a = np.exp(-theta)
    au = np.exp(-theta * u)
    av = np.exp(-theta * v)

    numerator = theta * (1 - a) * au * av
    denominator = (1 - a - (1 - au) * (1 - av)) ** 2

    return np.maximum(numerator / denominator, 1e-300)


def log_frank_density(theta, u, v):
    return np.log(frank_density(theta, u, v))


# ============================================================
# Kernels
# ============================================================

def uniform_kernel(t):
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) <= 1, 0.5, 0.0)


def biweight_kernel(t):
    t = np.asarray(t, dtype=float)
    return np.where(np.abs(t) <= 1, (15 / 16) * (1 - t**2) ** 2, 0.0)


def gaussian_kernel(t):
    t = np.asarray(t, dtype=float)
    return np.exp(-0.5 * t**2) / sqrt(2 * pi)


def get_kernel(kernel):
    if kernel == "uniform":
        return uniform_kernel
    if kernel == "biweight":
        return biweight_kernel
    if kernel == "gaussian":
        return gaussian_kernel
    raise ValueError("kernel must be 'uniform', 'biweight', or 'gaussian'.")


# ============================================================
# Bandwidth
# ============================================================

def kernel_moments(kernel):
    if kernel == "gaussian":
        R = 1 / (2 * sqrt(pi))  # int K^2
        mu2 = 1                 # int u^2 K(u)
    elif kernel == "biweight":
        R = 5 / 7
        mu2 = 1 / 7
    elif kernel == "uniform":
        R = 1 / 2
        mu2 = 1 / 3
    else:
        raise ValueError("unknown kernel")
    return R, mu2


def get_rule_of_thumb_bandwidth(samples, kernel="gaussian"):
    samples = np.asarray(samples, dtype=float)
    n = len(samples)

    sigma1 = np.std(samples[:, 0], ddof=1)
    sigma2 = np.std(samples[:, 1], ddof=1)
    rho = np.corrcoef(samples.T)[0, 1]

    term1, term2 = kernel_moments(kernel)

    common = (2 * sqrt(pi) * term1 / term2) ** (1 / 3)
    common *= (1 - rho**2) ** (5 / 12)
    common /= (1 + rho**2 / 2) ** (1 / 6)
    common /= n ** (1 / 6)

    return sigma1 * common, sigma2 * common


def get_bandwidth(data, band=None, kernel="gaussian"):
    if band is None:
        return get_rule_of_thumb_bandwidth(data, kernel=kernel)
    h1, h2 = band
    return float(h1), float(h2)


# ============================================================
# Local likelihood terms
# ============================================================

def local_weights(x, y, data, h1, h2, kernel="uniform"):
    data = np.asarray(data, dtype=float)
    K = get_kernel(kernel)

    u = data[:, 0]
    v = data[:, 1]

    wx = K((u - x) / h1) / h1
    wy = K((v - y) / h2) / h2

    return wx * wy


def rectangle_integral_frank(theta, x, y, h1, h2):
    """
    uniform kernel 専用の高速計算。
    K_h(t)=1/(2h) なので、長方形確率を 4h1h2 で割る。
    """
    u_low = max(x - h1, 0.0)
    u_high = min(x + h1, 1.0)
    v_low = max(y - h2, 0.0)
    v_high = min(y + h2, 1.0)

    prob = (
        frank_cdf(theta, u_high, v_high)
        - frank_cdf(theta, u_high, v_low)
        - frank_cdf(theta, u_low, v_high)
        + frank_cdf(theta, u_low, v_low)
    )

    return prob / (4 * h1 * h2)


def numerical_integral_frank(theta, x, y, h1, h2, kernel="biweight"):
    K = get_kernel(kernel)

    def integrand(v, u):
        wx = K((u - x) / h1) / h1
        wy = K((v - y) / h2) / h2
        return wx * wy * frank_density(theta, u, v)

    val, _ = integrate.dblquad(
        integrand,
        0.0,
        1.0,
        lambda u: 0.0,
        lambda u: 1.0,
    )

    return val


def integral_term_frank(theta, x, y, h1, h2, kernel="uniform"):
    if kernel == "uniform":
        return rectangle_integral_frank(theta, x, y, h1, h2)

    return numerical_integral_frank(theta, x, y, h1, h2, kernel=kernel)


def local_log_likelihood(
    theta,
    x,
    y,
    data,
    band=None,
    kernel="uniform",
):
    data = np.asarray(data, dtype=float)
    h1, h2 = get_bandwidth(data, band, kernel=kernel)

    weights = local_weights(
        x=x,
        y=y,
        data=data,
        h1=h1,
        h2=h2,
        kernel=kernel,
    )

    empirical_term = np.mean(
        weights * log_frank_density(theta, data[:, 0], data[:, 1])
    )

    correction_term = integral_term_frank(
        theta=theta,
        x=x,
        y=y,
        h1=h1,
        h2=h2,
        kernel=kernel,
    )

    return empirical_term - correction_term


# ============================================================
# Estimation
# ============================================================

def estimate_theta_grid(
    x,
    y,
    data,
    band=None,
    kernel="uniform",
    theta_min=-20.0,
    theta_max=20.0,
    step=0.1,
):
    theta_grid = np.arange(theta_min, theta_max + step, step)

    values = np.array([
        local_log_likelihood(
            theta=theta,
            x=x,
            y=y,
            data=data,
            band=band,
            kernel=kernel,
        )
        for theta in theta_grid
    ])

    return theta_grid[int(np.argmax(values))]


def estimate_theta_refined(
    x,
    y,
    data,
    band=None,
    kernel="uniform",
    theta_min=-20.0,
    theta_max=20.0,
    step=0.25,
):
    theta0 = estimate_theta_grid(
        x=x,
        y=y,
        data=data,
        band=band,
        kernel=kernel,
        theta_min=theta_min,
        theta_max=theta_max,
        step=step,
    )

    lower = max(theta_min, theta0 - step)
    upper = min(theta_max, theta0 + step)

    result = optimize.minimize_scalar(
        lambda theta: -local_log_likelihood(
            theta=theta,
            x=x,
            y=y,
            data=data,
            band=band,
            kernel=kernel,
        ),
        bounds=(lower, upper),
        method="bounded",
    )

    return result.x


def relative_local_dependence_frank_estimator(
    data,
    x,
    y,
    band=None,
    kernel_name="uniform",
    method="refined",
    theta_min=-20.0,
    theta_max=20.0,
    step=0.25,
):
    if method == "grid":
        theta_hat = estimate_theta_grid(
            x=x,
            y=y,
            data=data,
            band=band,
            kernel=kernel_name,
            theta_min=theta_min,
            theta_max=theta_max,
            step=step,
        )

    elif method == "refined":
        theta_hat = estimate_theta_refined(
            x=x,
            y=y,
            data=data,
            band=band,
            kernel=kernel_name,
            theta_min=theta_min,
            theta_max=theta_max,
            step=step,
        )

    else:
        raise ValueError("method must be 'grid' or 'refined'.")

    return 2 * theta_hat


def kernel_density_estimator(
    data,
    x,
    y,
    kernel_name="uniform",
    band=None,
):
    """
    2次元 KDE: f_hat(x, y)

    data: shape (n, 2) の array-like
    x, y: 評価点
    kernel_name: "uniform", "triangular", "epanechnikov", "gaussian"
    bandwidth: None, scalar, または (hx, hy)
    """
    data = np.asarray(data, dtype=float)

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("data must have shape (n, 2)")

    n = data.shape[0]
    if n == 0:
        raise ValueError("data must not be empty")
    
    hx, hy = band

    if hx <= 0 or hy <= 0:
        raise ValueError("bandwidth must be positive")

    ux = (x - data[:, 0]) / hx
    uy = (y - data[:, 1]) / hy

    _kernel = get_kernel(kernel_name)
    kx = _kernel(ux)
    ky = _kernel(uy)

    return np.mean(kx * ky) / (hx * hy)


def local_dependence_frank_estimator(
    data,
    x,
    y,
    kernel_name="uniform",
):
    from scipy.stats import rankdata
    from scipy.stats import norm
    pobs = np.column_stack([
        rankdata(data[:, j]) / (data.shape[0] + 1)
        for j in range(2)
    ])

    band = get_bandwidth(data=pobs, band=None, kernel=kernel_name)
    est1 = relative_local_dependence_frank_estimator(pobs, norm.cdf(x), norm.cdf(y), kernel_name=kernel_name, band=band, method="refined")

    band = get_bandwidth(data=data, band=None, kernel=kernel_name)
    est2 = kernel_density_estimator(data, x, y, kernel_name=kernel_name, band=band)
    return est1*est2