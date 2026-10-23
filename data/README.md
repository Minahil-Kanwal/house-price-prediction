# Data

The raw Ames Housing data is intentionally not committed to version control. It is downloaded from [OpenML dataset 42165](https://www.openml.org/d/42165) through the reproducible project command:

```powershell
python -m src.cli download
```

This creates `data/raw/ames_housing.csv`. Processed files, if created during analysis, belong in `data/processed/`. Do not use test-set-derived outputs as training inputs.

The source data is the Ames, Iowa housing dataset used by the House Prices prediction benchmark. Review the upstream data card and licensing terms before redistributing the raw data.
