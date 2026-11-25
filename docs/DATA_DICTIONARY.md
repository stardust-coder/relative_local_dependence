# Data dictionary

## `data/copula_grid_200.csv`
Each row corresponds to one copula, one replication, one bandwidth and one grid center (200,000 rows total). Columns: `family`, `rep`, `input` (the value `true` means direct copula observations; pandas may parse this as boolean), `h`, `u`, `v`, `interior`, `m` (number of local observations), `truth` (RLD), `LKT`, `LLB`, `LF`, `LF_success`, `LF_bound` (shape parameter near numerical bound).

## `data/sample_size_new_200.csv`
Simulation observations for `n=300` and `n=4800`, with `family,n,rep,h,u,v,m,truth,LKT,LLB,LF,LF_success,LF_bound`; combine with the `n=1200,h=0.25` rows from the main grid for the paper's table.

## `data/normality_all_samples.csv`
Per-replication point estimates and metadata; the original normality analysis standardizes each estimator by its empirical mean and empirical SD. It does not compare true-centered statistics to a derived asymptotic distribution.

## `data/bootstrap_61.npz`
Saved original and 61 bootstrap replicates for 3 maps: LLB-LD on original `(X,Y)`, LLB-LD on pseudo-observations `(U,V)`, LF on pseudo-observations; includes coordinate arrays. Bootstrap samples re-rank observations at each replicate.

## Border interpretation
At the full 25-point grid, some symmetric windows cross the copula boundary. The LF integration was clipped to the support but the rectangular LKT/LLB formulas were otherwise unchanged. These results are not all comparable under the same interior-point asymptotic theory.

## Naming convention

`LKT`, `LLB`, and `LF` are **RLD estimators** in the main tables. `LLB_LD` (rendered as **LLB-LD**) is the **un-normalized local dependence estimator** used for visualization in original and copula coordinates. In the rectangular-kernel experiments, `LLB_LD = 9*cov(S,T)/(h1*h2)`, whereas in the symmetric copula-window setting `LLB = 36*cov(S,T)/p_hat` estimates RLD. The archived main-grid CSV now uses `LF` and `LF_bound` as its column names.
