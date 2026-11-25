# Relative Local Dependence

Reproducible Python code and archived Monte Carlo outputs for the **LKT**, **LLB**, and **LF** estimators.

## How to cite

Please cite [our paper]().

```
@misc{sukeda2024relative,
      title={Relative local dependence of bivariate copulas}, 
      author={Issey Sukeda and Tomonari Sei},
      year={2024},
      eprint={2407.16948},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2407.16948}, 
}
```


## What is included?

| Experiment | Included original Monte Carlo data | Re-run script |
|---|---|---|
| Five copulas, **25 locations × 8 bandwidths × 200 replications**, `n=1200` | `data/copula_grid_200.csv` (200,000 rows) | `src/compare_three.py` |
| Sample sizes `n=300,1200,4800`, nine interior locations, `h=0.25` | `data/sample_size_new_200.csv` for 300 and 4800; 1200 from the main grid | `scripts/run_sample_size.py` |
| Normality diagnostics for `n=4800,30000,100000,400000` at `(0.5,0.5)` | `data/normality_all_samples.csv` | `scripts/run_normality.py` for new large samples |
| Bootstrap trinary dependence maps for `Y=X²+epsilon` | `data/bootstrap_61.npz` | `scripts/run_bootstrap_map.py` |

**Naming conventions:** The three estimators in the paper are always **LKT**, **LLB** and **LF**. In manuscript tables, **LLB estimates relative local dependence (RLD)** by dividing the Jones--Koch log-bilinear estimate of local dependence by the kernel density estimate. The **un-normalized local dependence function estimator** is called **LLB-LD** (`LLB_LD` in Python/CSV) and is used in the nonlinear-dependence maps. `LF` estimates RLD by local Frank likelihood fitting. Archived Monte Carlo CSV headers have also been standardized without changing any numerical values.

## Quick start

Tested in Python 3.13 with NumPy, SciPy, pandas, matplotlib. From the repository root:

```bash
python -m pip install -r requirements.txt
python scripts/make_tables.py
python scripts/make_normality_report.py
python scripts/plot_normality.py
python scripts/plot_bootstrap_map.py
python scripts/validate_outputs.py
```

The first command **does not run expensive simulations**: it computes the four Section 6 numerical summaries from archived Monte Carlo data, **saves CSV files and prints all summaries to the console**. It generates no LaTeX. Archived reference figures are in `figures/`; plotting scripts remain available for the exploratory dependence-map and QQ-plot experiments.

## Re-run Monte Carlo experiments (computationally expensive)

```bash
python src/compare_three.py --reps 200 --workers 6 --input true --starts 1 --out results/copula_grid
python scripts/run_sample_size.py --ns 300,4800 --reps 200 --workers 6 --out results/sample_size
python scripts/run_normality.py --ns 30000,100000,400000 --reps 150 --workers 6 --out results/normality
python scripts/run_bootstrap_map.py --reps 61 --workers 4 --out results/bootstrap_map
python scripts/plot_bootstrap_map.py --input results/bootstrap_map/bootstrap.npz --out results/bootstrap_map
```

The random-number seeds are set in the original simulation scripts. Floating-point output and optimizer convergence may depend slightly on software/platform versions. We supply archived raw outputs so the tables can be recreated without rerunning Monte Carlo.

## Numerical settings

Five copulas: Frank (3), Clayton (1), Gaussian (0.3), Gumbel (1.5), FGM (0.7). Main grid: `{0.1,0.3,0.5,0.7,0.9}²`; bandwidths `0.15,0.20,...,0.50`. Product rectangular kernel. LKT is scaled local Kendall's tau; LLB is the Jones–Koch type local moment estimator divided by the local density when targeting RLD; LF is the four-parameter local Frank likelihood estimator, with profiled intercept, bounded optimizer and `8×8` Gauss–Legendre quadrature. The full-grid results include **boundary-truncated windows** and should be interpreted as exploratory; interior-only results use symmetric windows.

### CSV summaries produced by `make_tables.py`

- `tables/bias_variance_rmse_h025_interior.csv`: Bias², variance, RMSE at `h=0.25`, nine interior locations.
- `tables/rmse_bandwidth_interior.csv`: RMSE at `h=0.15,0.20,0.25,0.30`, nine interior locations.
- `tables/rmse_sample_size.csv`: RMSE for `n=300,1200,4800`, `h=0.25`, nine interior locations.
- `tables/rmse_all25_every_bandwidth.csv`: RMSE across all 25 locations for all eight bandwidths, including `h=0.30`.

All four data frames are also printed to the terminal, with numeric results formatted to three decimal places. CSV files retain full numeric precision.

## Statistical caveats

* RLD and conventional local dependence are different targets; the original-scale LLB-LD map estimates classical local dependence, while the copula-space LF map estimates RLD.
* Bootstrap significance maps use 61 **pointwise** percentile intervals without multiplicity correction. A non-rejection is not proof of independence.
* Numerical bounds on LF shape parameters can affect performance, particularly near boundaries.
* Normal QQ plots and p-values are exploratory checks, **not** proofs of asymptotic normality.
* The included normality data for `n=4800` come from the fixed `h=0.25` sample-size experiment; the other sample sizes use bandwidth `0.25*(n/4800)**(-0.12)`.

## Publishing

No license has been chosen on the author's behalf. Before making a public GitHub repository, add the license and citation information you want to use. See `docs/DATA_DICTIONARY.md` for column details and notes about archived data.

## Estimator targets (important)

| Label | Python/CSV name | Estimand | Used in |
|---|---|---|---|
| LKT | `LKT` | RLD $r^c$ | Main simulations and paper tables |
| LLB | `LLB` | RLD $r^c = i^c/c$ | Main simulations and paper tables |
| LF | `LF` | RLD $r^c$ | Main simulations and paper tables, dependence maps |
| LLB-LD | `LLB_LD` | Classical local dependence $i^f$ or copula local dependence $i^c$ | Original-vs-pseudo and bootstrap maps only |

The final line is **not** a fourth RLD estimator. It is the *unnormalized* local-dependence estimate used as an intermediate quantity to construct LLB. In rectangular windows with equal bandwidth $h$, the estimators are `LLB_LD = 9*Cov(S,T)/h**2` and `LLB = 36*Cov(S,T)/p_hat`, where `p_hat=m/n`. These are different quantities.
