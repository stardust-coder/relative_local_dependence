import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from math import sqrt, pi, exp, log
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata
from scipy import integrate
import pandas as pd

def gaussian(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

def biweight_univariate_density(u):
    u = np.asarray(u)
    out = np.zeros_like(u, dtype=float)
    mask = np.abs(u) <= 1
    out[mask] = (15 / 16) * (1 - u[mask]**2)**2
    return out

def sample_frank_copula(theta, n, rng):
    """
    Sample (U, V) from Frank copula with parameter theta.

    Uses conditional inversion:
        V | U = u
    """
    if abs(theta) < 1e-10:
        return rng.uniform(size=(n, 2))

    u = rng.uniform(size=n)
    q = rng.uniform(size=n)

    A = np.exp(-theta * u)
    B = np.exp(-theta)

    # Solve conditional CDF inversion.
    # C_{2|1}(v|u) = q
    # Let E = exp(-theta v).
    # E = 1 + q * (B - 1) / (A - q * (A - 1))
    E = 1.0 + q * (B - 1.0) / (A - q * (A - 1.0))

    v = -np.log(E) / theta

    # Numerical safety
    u = np.clip(u, 1e-12, 1 - 1e-12)
    v = np.clip(v, 1e-12, 1 - 1e-12)

    return np.column_stack([u, v])


def sample_frank_normal_marginals(theta, n, rng):
    """
    Sample (X, Y) where:
        (U, V) ~ Frank copula(theta)
        X = Phi^{-1}(U), Y = Phi^{-1}(V)
    """
    uv = sample_frank_copula(theta, n, rng)
    xy = norm.ppf(uv)
    return xy


def true_joint_density_frank_normal(x, y, theta):
    """
    f_{X,Y}(x,y) = c_theta(Phi(x), Phi(y)) phi(x) phi(y)
    """
    u = norm.cdf(x)
    v = norm.cdf(y)

    from relative_local_dependence import frank_density
    c = frank_density(theta, u, v)
    return c * norm.pdf(x) * norm.pdf(y)


def true_local_dependence_frank_normal(x, y, theta):
    """
    For Frank copula, relative local dependence = 2 theta.

    Therefore:
        local dependence = 2 theta * f_{X,Y}(x,y)
    """
    fxy = true_joint_density_frank_normal(x, y, theta)
    return 2.0 * theta * fxy


def local_dependence_estimator(samples, x, y, kernel_name="biweight"):
    if kernel_name == "gaussian":
        kernel = gaussian
        term1 = 1 / (2 * np.sqrt(np.pi))
        term2 = 1
    elif kernel_name == "biweight":
        kernel = biweight_univariate_density
        term1 = 5 / 7
        term2 = 1 / 7
    else:
        raise ValueError("kernel_name must be 'gaussian' or 'biweight'")

    n = len(samples)
    X = samples[:, 0]
    Y = samples[:, 1]

    sigma1 = np.std(X, ddof=1)
    sigma2 = np.std(Y, ddof=1)
    rho_hat = np.corrcoef(X, Y)[0, 1]

    common = (2 * np.sqrt(np.pi) * term1 / term2) ** (1 / 3)
    common *= (1 - rho_hat**2) ** (5 / 12)
    common /= (1 + rho_hat**2 / 2) ** (1 / 6)
    common /= n ** (1 / 6)

    h1 = sigma1 * common
    h2 = sigma2 * common

    ux = (X - x) / h1
    uy = (Y - y) / h2

    w = kernel(ux) * kernel(uy) / (h1 * h2)

    g00 = np.mean(w)

    if g00 <= 1e-14:
        return np.nan

    g10 = np.mean(X * w)
    g01 = np.mean(Y * w)
    g11 = np.mean(X * Y * w)

    local_cov = (g11 - g10 * g01 / g00) / g00

    est = local_cov / (h1**2 * h2**2 * term2**2)

    return est

def true_local_dependence_gaussian(rho, sigma_x=1.0, sigma_y=1.0):
    return rho / (sigma_x * sigma_y * (1 - rho**2))


def run_asymptotic_experiment(
    estimator,
    ns=(500, 1000, 2000, 5000, 10000),
    R=300,
    rho=0.6,
    x0=0.0,
    y0=0.0,
    kernel_name="gaussian",
    seed=0,
):
    rng = np.random.default_rng(seed)

    #bivariate Gaussianからのサンプルを使う場合
    mean = np.array([0.0, 0.0])
    cov = np.array([
        [1,rho],
        [rho,1],
    ])
    true_val = true_local_dependence_gaussian(rho, 1, 1) #定数

    # #Frank copula + standard normalからのサンプルを使う場合
    # true_val = true_local_dependence_frank_normal(
    #     x=x0,
    #     y=y0,
    #     theta=rho,
    # )

    rows = []

    for n in ns:
        print("Testing with #Samples=", n, "...")

        use_joblib = True  # False にすると通常の for ループ

        if use_joblib:
            from joblib import Parallel, delayed
            seed_seq = np.random.SeedSequence(seed)
            child_seeds = seed_seq.spawn(R)

            def run_one(child_seed):
                local_rng = np.random.default_rng(child_seed)
                
                #Gaussianの場合
                samples = local_rng.multivariate_normal(mean, cov, size=n)
                
                # #Frankの場合
                # samples = sample_frank_normal_marginals(
                #     theta=rho,
                #     n=n,
                #     rng=local_rng,
                # )
                return estimator(
                    samples, x0, y0, kernel_name=kernel_name
                )

            estimates = Parallel(n_jobs=-1)(
                delayed(run_one)(child_seeds[i])
                for i in tqdm(range(R))
            )

        else:
            estimates = []

            for _ in tqdm(range(R)):
                samples = rng.multivariate_normal(mean, cov, size=n)

                est = estimator(
                    samples, x0, y0, kernel_name=kernel_name
                )

                estimates.append(est)

        estimates = np.array(estimates)

        bias = np.mean(estimates) - true_val
        abs_bias = abs(bias)
        variance = np.var(estimates, ddof=1)
        mse = np.mean((estimates - true_val) ** 2)

        rows.append({
            "n": n,
            "mean_est": np.mean(estimates),
            "true": true_val,
            "bias": bias,
            "abs_bias": abs_bias,
            "variance": variance,
            "mse": mse,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from relative_local_dependence import local_dependence_frank_estimator
    x0,y0 = 0,0
    R = 3000
    df = run_asymptotic_experiment(
        estimator=local_dependence_frank_estimator,
        R = R,
        ns = [1000, 2000, 5000, 10000, 20000, 50000],
        rho=0.6,
        kernel_name="gaussian",
        x0=x0,
        y0=y0
    )

    print(df)

    def loglog_slope(df, ycol):
        x = np.log(df["n"].values)
        y = np.log(df[ycol].values)

        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    def add_reference_line(df, ycol, slope, label):
        n = df["n"].values
        y = df[ycol].values

        c = y[0] / (n[0] ** slope)
        ref = c * n ** slope

        plt.loglog(n, ref, linestyle="--", label=label)
    
    bias_slope, _ = loglog_slope(df, "abs_bias")
    var_slope, _ = loglog_slope(df, "variance")
    mse_slope, _ = loglog_slope(df, "mse")

    print("bias slope:", bias_slope)
    print("variance slope:", var_slope)
    print("mse slope:", mse_slope)

    from pathlib import Path
    save_dir = f"plot/{x0}-{y0}-{R}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.loglog(df["n"], df["abs_bias"], marker="o", label="absolute bias")
    add_reference_line(df, "abs_bias", slope=-1/3, label="reference n^(-1/3)")
    plt.xlabel("n")
    plt.ylabel("absolute bias")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/1.png")
    plt.close()

    plt.figure()
    plt.loglog(df["n"], df["variance"], marker="o", label="variance")
    add_reference_line(df, "variance", slope=-2/3, label="reference n^(-2/3)")
    plt.xlabel("n")
    plt.ylabel("variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/2.png")
    plt.close()

    plt.figure()
    plt.loglog(df["n"], df["mse"], marker="o", label="MSE")
    add_reference_line(df, "mse", slope=-2/3, label="reference n^(-2/3)")
    plt.xlabel("n")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/3.png")
    plt.close()