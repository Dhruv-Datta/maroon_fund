# Stock-specific model artifacts

This folder holds **per-ticker model artifacts** and **archived per-ticker pipelines**. It is **not used by the main pipeline** (1_buyside_model, 2_sellside_model, or combined_buy_sell_model). The combined trainer’s step that would copy models here is commented out in code.

## Contents

- **amat/** – Per-ticker buy and sell models for AMAT: `amat_buy_model.joblib`, `amat_sell_model.joblib`, `utils.csv`.
- **lmt/** – Per-ticker buy and sell models for LMT: `LMT_buy_model.joblib`, `LMT_sell_model.joblib`, `utils.csv`.
- **archived_models/** – Old self-contained pipelines for specific tickers:
  - **amzn_buyside_model**, **meta_buyside_model**, **nvda_buyside_model** – Buyside-style scripts (manual input, train, test) and joblibs.
  - **amzn_sellside_model**, **nvda_sellside_model** – Sellside-style scripts and sell-signal CSVs/joblibs.

Use this folder as reference or backup only; the active workflows use the top-level buyside, sellside, and combined folders.
