# local Frank vs local log-bilinear

This bundle contains all Python code needed to **regenerate the Monte Carlo results from scratch**, without existing CSVs or access to GitHub. The simulation uses copula observations directly (no rank transformation), product rectangle kernels, identical observations for both methods, 30 replications, 25 points and `h=0.15,0.20,...,0.50`.

## Setup

Python 3.10+ recommended. In this directory:

```bash
python -m pip install -r requirements.txt
python run_comparison.py --output results
```

The default full simulation performs 5 x 30 x 25 x 8 = 30,000 paired estimation settings; it may take a substantial amount of CPU time. Fast smoke test:

```bash
python run_comparison.py --families 'Frank(3)' --reps 1 --bandwidths 0.25 --axes 0.5 --output smoke
```

Outputs: `raw.csv`, `summary.csv`, `spatial.csv`, `rmse.png`.

## Files
- `run_comparison.py`: standalone study driver and summaries.
- `local_frank_fit.py`: local Frank objective and optimization.
- `jones_koch_moment.py`: rectangle-weighted moment estimator, *named local log-bilinear in tables*.
- `local_frank_spatial.py`: copula generation dispatch, target function for Gumbel.
- `local_frank_simulation.py`: copula generation and ground-truth RLD for remaining families.
- `requirements.txt`: runtime dependencies.

This is a new standalone runner built from the prior implementations, not the original `add_005_bandwidths.py` patch script, which depended on older CSVs. Minor floating-point differences may occur across platform versions. The Frank optimizer still uses a fixed coarse grid to locate candidate maxima followed by continuous refinement; it never uses the true parameter to select its range. This is a finite-sample comparison, not a general consistency guarantee.
