# Bengaluru Air Quality Analysis

Statistical Analysis and Predictive Modelling of Air Quality in Bengaluru using
Classical Regression and Machine Learning Techniques (AAT project).

Target variable: **PM2.5 (µg/m³)** · Source: KSPCB / CPCB hourly station reports
collected from <https://data.opencity.in/dataset/bengaluru-hourly-air-quality-reports>.

## Interactive dashboard (Phase 1 + Phase 2)

```bash
git clone https://github.com/suprajavrao/Air-Quality-Index-Analysis.git

cd Air-Quality-Index-Analysis

pip install -r requirements_gui.txt

streamlit run app.py
```

Then open <http://localhost:8501> (Streamlit prints the exact URL/port).

Sidebar sections:

| Section | Contents |
|---|---|
| Overview | Descriptive statistics, correlation matrix, target distribution |
| Phase 1 · Classical Regression | SLR (PM2.5 ~ PM10), MLR equation, coefficient table with t/p, per-predictor contribution breakdown and waterfall, predicted-vs-actual |
| Phase 1 · Assumption Testing | VIF/tolerance, Breusch–Pagan (+ HC3 robust SEs), Jarque–Bera, Shapiro–Wilk, Durbin–Watson, residual/Q-Q/scale-location plots, consolidated table |
| Phase 1 · Bootstrap Validation | Resampling with replacement, empirical coefficient distributions, normal overlays, bootstrap vs theoretical SE |
| Phase 2 · Feature Selection | Forward selection and backward elimination by AIC, Ridge and LASSO coefficient paths |
| Phase 2 · ML Models | Decision Tree, Random Forest, SVR (RBF), XGBoost, feature importance |
| Phase 2 · Comparison & Selection | 5-fold cross-validation, held-out test metrics, final model choice |

Controls in the sidebar: section selector, number of rows used, and the
predictor set. Every displayed number is computed live from the processed data.

## Command-line analysis

```bash
cd src
python analyze_small.py        # end-to-end analysis, writes results/ and reports/
```

## Rebuilding the modelling extract

`app.py` reads `data/processed/model_sample.csv`, a compact extract built with the
standard library only:

```bash
python src/make_model_sample.py
```

Raw station files stay in `data/raw/` (`blr-*.csv`, one per station).

## Notes on this machine

The host has roughly 0.6–0.9 GB of free RAM, which caused `MemoryError` failures
when pandas/statsmodels were imported inside the Streamlit runtime. For that
reason:

- `app.py` uses **NumPy/SciPy/SciKit** only; the OLS inference (SEs, t, p, F, AIC,
  BIC, VIF, Breusch–Pagan, Jarque–Bera, Durbin–Watson, HC3 errors) is implemented
  directly in `app.py`.
- Models run on a 1,000–6,000 row sample; SVR is fitted on a 1,500-row subsample
  because its kernel matrix grows quadratically.
- `requirements_gui.txt` pins the verified working versions (NumPy 1.26.4 with
  SciPy 1.14.1 / SciKit 1.5.2 — NumPy 2.x breaks that binary pairing here).

`dashboard.html` is a dependency-free static summary that opens directly in a
browser if Python is unavailable.
# Air-Quality-Index
