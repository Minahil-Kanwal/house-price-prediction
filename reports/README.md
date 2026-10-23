# Reports

Training writes reviewable artifacts here:

- `results/data_validation.json` — schema, completeness, duplication, sparsity, and validity checks.
- `results/model_comparison.csv` — holdout and cross-validation results for all baseline models.
- `results/error_analysis.csv` — residuals ordered by absolute error.
- `figures/` — actual-vs-predicted, residual, and feature-importance plots when supported.

These are generated outputs and intentionally excluded from source control so that each run remains tied to its own local data and environment.
