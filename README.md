# Production-Ready House Price Prediction

An end-to-end regression system for estimating residential sale prices from the Ames Housing dataset. The repository is designed as a portfolio-quality ML service: data validation, reproducible model selection, leakage-safe preprocessing, model diagnostics, a documented FastAPI service, and a Streamlit interface are all built around one serialized inference pipeline.

## Business problem

Accurate price estimates help sellers, buyers, and analysts compare similar properties and identify the attributes most associated with sale value. This project predicts `SalePrice` in USD from property size, condition, age, amenities, location, and sale context. It is a decision-support estimate—not a formal appraisal.

## Objectives

- Validate and profile raw data before training.
- Compare linear, regularized, and tree-based regressors with consistent metrics.
- Prevent data leakage by placing feature engineering, imputation, encoding, and scaling inside sklearn pipelines.
- Tune the cross-validation winner using training data only; reserve the test partition for final evaluation.
- Serve the exact trained pipeline through FastAPI and Streamlit without retraining.

## Dataset

The project obtains the canonical Ames Housing / House Prices training schema from [OpenML dataset 42165](https://www.openml.org/d/42165) on demand. The dataset is not committed to the repository. `data/raw/ames_housing.csv` contains the locally cached input and is ignored by Git.

The model accepts a curated set of 32 explainable attributes rather than requiring every competition column. It includes living area, quality scores, year fields, bathrooms, garage capacity, exterior and kitchen quality, neighborhood, and related property attributes. Seven domain-driven features are calculated inside the inference pipeline: total interior area, total bathrooms, outdoor area, house/remodel age at sale, and garage/basement indicators.

## Project structure

```text
.
├── api/                  # FastAPI application, routes, schema, predictor adapter
├── data/                 # Ignored raw/processed data plus acquisition notes
├── models/               # Ignored serialized pipeline and generated metadata
├── notebooks/            # EDA and experiment narrative for VS Code Jupyter
├── reports/              # Generated validation, evaluation, and diagnostic outputs
├── scripts/              # Optional Windows training convenience script
├── src/                  # Reusable ML/data modules and command-line workflow
├── streamlit_app/        # Non-technical local prediction interface
├── tests/                # Focused regression and API-contract tests
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.12 is recommended. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, call `.venv\Scripts\python.exe` in the commands below instead. The optional `scripts/run_training.ps1` performs the setup and training steps in one command.

## Train the model

Download data, run validation, compare all baselines, tune the CV winner, and save the model:

```powershell
python -m src.cli train --download
```

To download or validate separately:

```powershell
python -m src.cli download
python -m src.cli validate
```

The complete pipeline is saved at `models/house_price_pipeline.joblib`; metadata is saved at `models/model_metadata.json`. Generated figures and CSV/JSON reports are written under `reports/`.

### Modeling approach

The workflow validates missingness, duplicate records/IDs, empty strings, constant and near-constant columns, sparsity, and basic invalid values. Missing numeric values are median-imputed and categorical values use their most frequent observed category—both learned exclusively on each training fold. One-hot encoding ignores unseen production categories safely. Robust scaling is used for coefficient-based models only; tree models are intentionally not scaled.

The comparison includes Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, and Gradient Boosting. Selection is based on mean five-fold cross-validation RMSE on the training partition. `RandomizedSearchCV` tunes only that winner. Final MAE, MSE, RMSE, and R² are calculated on the untouched test partition. The generated reports document model stability, errors, residuals, and feature importance where the winning estimator supports it.

## Run the API

Train first, then start the service:

```powershell
uvicorn api.main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive OpenAPI documentation.

Endpoints:

- `GET /` — service information
- `GET /health` — readiness and model-artifact status
- `POST /predict` — validated property attributes and predicted price

Example request:

```powershell
$body = @{
  OverallQual = 7; OverallCond = 5; GrLivArea = 1710; TotalBsmtSF = 856
  '1stFlrSF' = 856; '2ndFlrSF' = 854; FullBath = 2; HalfBath = 1
  BsmtFullBath = 1; BsmtHalfBath = 0; BedroomAbvGr = 3; TotRmsAbvGrd = 8
  YearBuilt = 2003; YearRemodAdd = 2003; YrSold = 2008; GarageCars = 2
  GarageArea = 548; WoodDeckSF = 0; OpenPorchSF = 61; LotArea = 8450
  MSZoning = 'RL'; Neighborhood = 'CollgCr'; BldgType = '1Fam'; HouseStyle = '2Story'
  ExterQual = 'Gd'; KitchenQual = 'Gd'; HeatingQC = 'Ex'; BsmtQual = 'Gd'
  GarageFinish = 'RFn'; SaleCondition = 'Normal'; CentralAir = 'Y'; LotShape = 'Reg'
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body
```

The Pydantic contract rejects missing, malformed, out-of-range, and extra fields. The service returns safe error messages and never exposes tracebacks to callers.

## Run the Streamlit application

```powershell
streamlit run streamlit_app/app.py
```

The interface uses friendly labels and defaults, validates the same Pydantic contract as the API, and shows a clearly formatted USD estimate.

## Notebook

Open [notebooks/house_price_eda.ipynb](notebooks/house_price_eda.ipynb) in VS Code’s Jupyter support. It is intentionally limited to inspection, EDA, visualizations, experiment review, and conclusions; reusable production logic lives in `src/`.

## Verification

```powershell
pytest -q
python -m src.cli validate
```

After training, use `GET /health`, the Swagger `/docs` page, and the Streamlit form to confirm that model loading and prediction work end to end.

## Screenshots

After running the local UI, save an API Docs screenshot and a Streamlit prediction screenshot under `docs/images/`, then embed them here using descriptive alt text. Generated diagnostic figures already appear under `reports/figures/` after training.

## Technologies

Python, pandas, NumPy, scikit-learn, Joblib, FastAPI, Pydantic, Uvicorn, Streamlit, Matplotlib, Seaborn, pytest, and VS Code Jupyter.

## Limitations and future improvements

- The Ames dataset represents one city and historical sales; it should not be extrapolated to other markets without retraining.
- Results can be affected by unobserved factors such as interior condition, renovations, economic conditions, and buyer demand.
- Future work could add data/version tracking, experiment tracking, drift monitoring, containerization, CI, authenticated model serving, and calibrated prediction intervals.

## License and author

Released under the [MIT License](LICENSE). Built as an ML engineering portfolio project.
