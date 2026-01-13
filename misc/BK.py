import numpy as np

def _silverman_bw(z: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for 1D data."""
    z = np.asarray(z, dtype=float)
    n = z.size
    if n < 2:
        return 1.0
    std = np.std(z, ddof=1)
    if std == 0:
        return 1.0
    return 1.06 * std * (n ** (-1/5))

def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u * u)

def _nw_conditional_mean(x: np.ndarray, y: np.ndarray, y0, bw: float) -> np.ndarray:
    """
    Nadaraya–Watson estimate of E[x | y = y0] with Gaussian kernel.
    x, y are 1D arrays of equal length.
    y0 can be scalar or array-like.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    y0 = np.asarray(y0, dtype=float)

    if bw <= 0:
        raise ValueError("Bandwidth bw must be > 0")

    # weights: shape (n, m) where m = len(y0)
    u = (y[:, None] - y0.reshape(1, -1)) / bw
    w = _gaussian_kernel(u)

    denom = w.sum(axis=0)
    # avoid divide-by-zero in extreme cases
    denom = np.where(denom == 0, np.nan, denom)

    num = (x[:, None] * w).sum(axis=0)
    return num / denom

def local_dependence_H(
    X, Y, x0, y0, *,
    bw_y=None, bw_x=None,
    eps=1e-12
):
    """
    Bairamov & Kotz local dependence function H(x,y) (sample-based estimator).

    Definition:
      H(x,y) = (rho_{X,Y} + phi_X(y)*phi_Y(x)) / (sqrt(1+phi_Y(x)^2)*sqrt(1+phi_X(y)^2))
      phi_X(y) = (E[X] - E[X|Y=y]) / sigma_X
      phi_Y(x) = (E[Y] - E[Y|X=x]) / sigma_Y

    Inputs:
      X, Y : 1D samples (same length)
      x0, y0 : evaluation points (scalar or array-like; vectorized)
      bw_y : bandwidth for estimating E[X|Y=y]  (defaults to Silverman on Y)
      bw_x : bandwidth for estimating E[Y|X=x]  (defaults to Silverman on X)

    Returns:
      H values with shape broadcastable to (len(x0), len(y0)) if both arrays,
      but by default returns array aligned to y0 for phi_X and to x0 for phi_Y.
      If both x0 and y0 are arrays, returns 2D grid H[i,j] = H(x0[i], y0[j]).
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    if X.size != Y.size:
        raise ValueError("X and Y must have the same length.")

    n = X.size
    if n < 2:
        raise ValueError("Need at least 2 samples.")

    EX = np.mean(X)
    EY = np.mean(Y)

    varX = np.var(X, ddof=1)
    varY = np.var(Y, ddof=1)
    if varX <= 0 or varY <= 0:
        raise ValueError("X and Y must have positive variance.")

    sigmaX = np.sqrt(varX)
    sigmaY = np.sqrt(varY)

    # Pearson correlation
    covXY = np.cov(X, Y, ddof=1)[0, 1]
    rho = covXY / (sigmaX * sigmaY)

    # Bandwidths
    if bw_y is None:
        bw_y = _silverman_bw(Y)
    if bw_x is None:
        bw_x = _silverman_bw(X)
    bw_y = max(float(bw_y), eps)
    bw_x = max(float(bw_x), eps)

    x0 = np.asarray(x0, dtype=float)
    y0 = np.asarray(y0, dtype=float)

    # Estimate conditional means
    # E[X | Y = y0] for each y0
    EX_given_Y = _nw_conditional_mean(X, Y, y0.ravel(), bw=bw_y)  # shape (m,)
    # E[Y | X = x0] for each x0
    EY_given_X = _nw_conditional_mean(Y, X, x0.ravel(), bw=bw_x)  # shape (k,)

    phiX = (EX - EX_given_Y) / sigmaX  # shape (m,)
    phiY = (EY - EY_given_X) / sigmaY  # shape (k,)

    # Build H on grid if both x0 and y0 are arrays
    phiY_grid = phiY.reshape(-1, 1)                 # (k,1)
    phiX_grid = phiX.reshape(1, -1)                 # (1,m)

    numerator = rho + phiX_grid * phiY_grid
    denom = np.sqrt(1.0 + phiY_grid**2) * np.sqrt(1.0 + phiX_grid**2)
    H_grid = numerator / denom

    # If original x0/y0 were scalars, return scalar
    if x0.ndim == 0 and y0.ndim == 0:
        return float(H_grid[0, 0])

    # If one scalar, return 1D aligned with the other
    if x0.ndim == 0 and y0.ndim != 0:
        return H_grid[0, :]  # over y0
    if x0.ndim != 0 and y0.ndim == 0:
        return H_grid[:, 0]  # over x0

    # Both arrays -> 2D grid H[x_index, y_index]
    return H_grid


# -----------------------
# Example
# -----------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    n = 2000
    X = rng.normal(size=n)
    Y = 0.7 * X + rng.normal(scale=0.7, size=n)
    plt.figure(figsize=(5,5))
    plt.scatter(X,Y)
    plt.savefig("BK_data.png")
    print("H(0,0) =", local_dependence_H(X, Y, 0.0, 0.0))

    # 評価グリッド
    xs = np.linspace(-2.5, 2.5, 80)
    ys = np.linspace(-2.5, 2.5, 80)

    # グリッド評価
    H = local_dependence_H(X, Y, xs, ys)  # shape = (len(xs), len(ys))
    print("H grid shape:", H.shape)

    # ---- 可視化 ----
    plt.figure()
    im = plt.imshow(
        H.T,
        origin="lower",
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        aspect="auto"
    )
    plt.colorbar(im, label="H(x, y)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Local Dependence Function H(x, y)")
    plt.show()
    plt.savefig("BK_estimate.png")
