"""Summarize archived Monte Carlo results as CSV and console tables (no LaTeX).

Run: python scripts/make_tables.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tables'
OUT.mkdir(exist_ok=True)
METHODS = ('LKT', 'LLB', 'LF')
FAMILIES = ('Frank(3)', 'Clayton(1)', 'Gaussian(0.3)', 'Gumbel(1.5)', 'FGM(0.7)')

main = pd.read_csv(ROOT / 'data/copula_grid_200.csv')
assert {'LKT', 'LLB', 'LF', 'LF_success', 'LF_bound'} <= set(main.columns)
assert len(main) == 200_000, f'Expected 200,000 records, got {len(main)}'
assert main.groupby(['family', 'h']).size().eq(25 * 200).all()


def components(data, method):
    """Equal-weight center-wise squared bias, variance and pooled RMSE."""
    w = data[np.isfinite(data[method])]
    groups = list(w.groupby(['u', 'v'], sort=False))
    assert groups, 'No valid observations'
    assert len({len(g) for _, g in groups}) == 1, 'Unequal replication counts across centers'
    bias2 = np.mean([(g[method].mean() - g.truth.iloc[0]) ** 2 for _, g in groups])
    variance = np.mean([np.mean((g[method] - g[method].mean()) ** 2) for _, g in groups])
    rmse = np.sqrt(np.mean((w[method] - w.truth) ** 2))
    assert np.isclose(bias2 + variance, rmse ** 2, atol=1e-8)
    return bias2, variance, rmse


def save_and_print(frame, filename, title):
    path = OUT / filename
    frame.to_csv(path, index=False)
    print(f'\n{title}\n' + '-' * len(title))
    print(frame.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print(f'Saved: {path.relative_to(ROOT)}')


interior = main[main.u.isin([0.3, 0.5, 0.7]) & main.v.isin([0.3, 0.5, 0.7])]

# Squared bias / variance / RMSE at h = 0.25: same nine interior centers.
rows = []
for family in FAMILIES:
    g = interior[(interior.family == family) & (interior.h == 0.25)]
    assert len(g) == 9 * 200
    metrics = {m: components(g, m) for m in METHODS}
    for k, metric in enumerate(('Squared bias', 'Variance', 'RMSE')):
        rows.append(dict(family=family, h=0.25, metric=metric,
                         **{m: metrics[m][k] for m in METHODS}))
summary = pd.DataFrame(rows)
save_and_print(summary, 'bias_variance_rmse_h025_interior.csv',
               'Squared bias / variance / RMSE: h=0.25, 9 interior points')

# Bandwidth effect with nine fixed interior centers.
rows = []
for family in FAMILIES:
    for h in (0.15, 0.20, 0.25, 0.30):
        g = interior[(interior.family == family) & (interior.h == h)]
        assert len(g) == 9 * 200
        rows.append(dict(family=family, h=h,
                         **{m: components(g, m)[2] for m in METHODS}))
bw = pd.DataFrame(rows)
save_and_print(bw, 'rmse_bandwidth_interior.csv',
               'RMSE by bandwidth: 9 interior points')

# All 25 centers, including boundary-truncated windows, all eight bandwidths.
rows = []
for family in FAMILIES:
    for h in sorted(main.h.unique()):
        g = main[(main.family == family) & (main.h == h)]
        assert len(g) == 25 * 200
        rows.append(dict(family=family, h=h,
                         **{m: components(g, m)[2] for m in METHODS}))
all25 = pd.DataFrame(rows)
save_and_print(all25, 'rmse_all25_every_bandwidth.csv',
               'RMSE: full 25-point grid, all bandwidths (boundary-affected)')
print('\nRMSE: full 25-point grid at h=0.30')
print(all25[all25.h == 0.30].to_string(index=False, float_format=lambda x: f'{x:.3f}'))

# Sample size effect, retaining archived n=1200 data from the main experiment.
extra = pd.read_csv(ROOT / 'data/sample_size_new_200.csv')
base = interior[interior.h == 0.25].copy()
base['n'] = 1200
cols = ['family', 'n', 'rep', 'h', 'u', 'v', 'm', 'truth', 'LKT', 'LLB', 'LF', 'LF_success', 'LF_bound']
assert all(c in extra.columns and c in base.columns for c in cols)
combined = pd.concat([extra[cols], base[cols]], ignore_index=True)
rows = []
for family in FAMILIES:
    for n in (300, 1200, 4800):
        g = combined[(combined.family == family) & (combined.n == n)]
        assert len(g) == 9 * 200, (family, n, len(g))
        rows.append(dict(family=family, n=n,
                         **{m: components(g, m)[2] for m in METHODS}))
sizes = pd.DataFrame(rows)
save_and_print(sizes, 'rmse_sample_size.csv', 'RMSE by sample size: h=0.25, 9 interior points')

print('\nCompleted: four CSV summaries; no LaTeX files generated.')
