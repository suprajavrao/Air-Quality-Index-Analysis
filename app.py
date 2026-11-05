"""
Streamlit dashboard — Bengaluru air-quality regression study (Phase 1 & 2).

Deliberately pandas-free and statsmodels-free: this host has very little free
RAM, so the OLS diagnostics are computed directly with NumPy/SciPy.

Run from the AirQuality_Bengaluru_Analysis directory:
    streamlit run app.py
"""

import csv
import os
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Bengaluru Air Quality Dashboard",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(BASE, "data", "processed", "model_sample.csv"),
    os.path.join(BASE, "data", "processed", "combined_raw_sample.csv"),
]

FIELDS = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "Ozone",
          "Temperature", "RelativeHumidity", "WindSpeed"]

LABELS = {
    "PM2.5": "PM2.5 (µg/m³)",
    "PM10": "PM10 (µg/m³)",
    "NO": "NO (µg/m³)",
    "NO2": "NO2 (µg/m³)",
    "NH3": "NH3 (µg/m³)",
    "SO2": "SO2 (µg/m³)",
    "CO": "CO (mg/m³)",
    "Ozone": "Ozone (µg/m³)",
    "Temperature": "Temperature (°C)",
    "RelativeHumidity": "Relative humidity (%)",
    "WindSpeed": "Wind speed (m/s)",
}


# --------------------------------------------------------------------- data io
@st.cache_data(show_spinner=False)
def load_matrix(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        try:
            idx = [header.index(c) for c in FIELDS]
        except ValueError as exc:
            raise ValueError(f"missing column in {os.path.basename(path)}: {exc}")
        rows = []
        for row in reader:
            if len(row) <= max(idx):
                continue
            record = []
            for i in idx:
                cell = row[i].strip()
                record.append(float(cell) if cell not in ("", "nan", "NaN") else np.nan)
            rows.append(record)
    return np.array(rows, dtype=float)


def load_data():
    for path in CANDIDATES:
        if os.path.exists(path):
            return load_matrix(path), path
    raise FileNotFoundError(
        "No processed dataset found. Run src/make_model_sample.py first."
    )


# ------------------------------------------------------------------ OLS engine
class OLSFit:
    """Plain OLS with the usual inference quantities (X must include intercept)."""

    def __init__(self, X, y):
        self.X, self.y = X, y
        n, k = X.shape
        self.n, self.k = n, k
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.beta = XtX_inv @ X.T @ y
        self.fitted = X @ self.beta
        self.resid = y - self.fitted
        sse = float(self.resid @ self.resid)
        df_resid = max(n - k, 1)
        sigma2 = sse / df_resid
        self.cov = sigma2 * XtX_inv
        self.se = np.sqrt(np.clip(np.diag(self.cov), 1e-30, None))
        self.tvalues = self.beta / self.se
        self.pvalues = 2 * stats.t.sf(np.abs(self.tvalues), df_resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1 - sse / ss_tot if ss_tot > 0 else np.nan
        self.r2_adj = 1 - (1 - self.r2) * (n - 1) / df_resid
        self.sse, self.ss_tot, self.df_resid = sse, ss_tot, df_resid
        if k > 1 and sigma2 > 0:
            self.fvalue = ((ss_tot - sse) / (k - 1)) / sigma2
            self.f_pvalue = stats.f.sf(self.fvalue, k - 1, df_resid)
        else:
            self.fvalue, self.f_pvalue = np.nan, np.nan
        llf = -0.5 * n * (np.log(2 * np.pi) + np.log(max(sse / n, 1e-30)) + 1)
        self.aic = -2 * llf + 2 * k
        self.bic = -2 * llf + k * np.log(n)

    def robust_se(self):
        """HC3 heteroskedasticity-consistent standard errors."""
        X = self.X
        h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
        w = (self.resid ** 2) / np.clip((1 - h) ** 2, 1e-12, None)
        meat = X.T @ (X * w[:, None])
        cov = np.linalg.pinv(X.T @ X) @ meat @ np.linalg.pinv(X.T @ X)
        return np.sqrt(np.clip(np.diag(cov), 1e-30, None))


def design(X):
    return np.column_stack([np.ones(len(X)), X])


def vif_values(X):
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        fit = OLSFit(design(others), X[:, j])
        out.append(np.inf if fit.r2 >= 1 else 1.0 / (1.0 - fit.r2))
    return np.array(out)


def breusch_pagan(fit, X):
    e2 = fit.resid ** 2
    aux = OLSFit(design(X), e2)
    lm = fit.n * aux.r2
    return lm, float(stats.chi2.sf(lm, X.shape[1]))


def jarque_bera(resid):
    n = len(resid)
    s = float(stats.skew(resid))
    k = float(stats.kurtosis(resid, fisher=False))
    jb = n / 6.0 * (s ** 2 + (k - 3) ** 2 / 4.0)
    return jb, float(stats.chi2.sf(jb, 2)), s, k


def durbin_watson(resid):
    return float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))


def metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
            float(mean_absolute_error(y_true, y_pred)))


# --------------------------------------------------------------------- helpers
def fig_show(fig):
    st.pyplot(fig)
    plt.close(fig)


def aic_of(X, y):
    return OLSFit(design(X), y).aic


def forward_selection(X, y, names):
    remaining, selected, history = list(range(X.shape[1])), [], []
    best = np.inf
    while remaining:
        scored = sorted(
            ((aic_of(X[:, selected + [j]], y), j) for j in remaining),
            key=lambda t: t[0],
        )
        if scored[0][0] >= best - 1e-9:
            break
        best = scored[0][0]
        selected.append(scored[0][1])
        remaining.remove(scored[0][1])
        history.append({"Step": len(selected), "Added": names[selected[-1]],
                        "AIC": round(best, 3)})
    return [names[j] for j in selected], history


def backward_elimination(X, y, names):
    selected, history = list(range(X.shape[1])), []
    best = aic_of(X, y)
    while len(selected) > 1:
        scored = sorted(
            ((aic_of(X[:, [s for s in selected if s != j]], y), j)
             for j in selected),
            key=lambda t: t[0],
        )
        if scored[0][0] >= best - 1e-9:
            break
        best = scored[0][0]
        selected.remove(scored[0][1])
        history.append({"Step": len(selected), "Dropped": names[scored[0][1]],
                        "AIC": round(best, 3)})
    return [names[j] for j in selected], history


st.markdown(
    "<h1 style='text-align:center;color:#1f77b4;margin-bottom:0'>"
    "🌬️ Bengaluru Air Quality Analysis Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:#555'>Statistical Analysis and Predictive "
    "Modelling of Air Quality in Bengaluru using Classical Regression and "
    "Machine Learning Techniques</p>",
    unsafe_allow_html=True,
)

try:
    MATRIX, SOURCE = load_data()
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

TARGET = FIELDS.index("PM2.5")
AVAILABLE = [FIELDS[j] for j in range(1, len(FIELDS))
             if np.isfinite(MATRIX[:, j]).sum() > 0.5 * len(MATRIX)]

st.sidebar.header("Controls")
page = st.sidebar.radio(
    "Section",
    [
        "📊 Overview",
        "🔬 Phase 1 · Classical Regression",
        "✅ Phase 1 · Assumption Testing",
        "🎲 Phase 1 · Bootstrap Validation",
        "🎯 Phase 2 · Feature Selection",
        "🤖 Phase 2 · ML Models",
        "🏆 Phase 2 · Comparison & Selection",
    ],
)
n_rows = st.sidebar.slider("Rows used", 1000, 6000, 4000, step=500)
predictors = st.sidebar.multiselect(
    "Predictors", AVAILABLE,
    default=[p for p in ["PM10", "NO", "NO2", "SO2", "CO", "Ozone"] if p in AVAILABLE],
)
st.sidebar.caption(f"Target: **PM2.5** · source: `{os.path.basename(SOURCE)}`")

if not predictors:
    st.warning("Select at least one predictor in the sidebar.")
    st.stop()

# ------------------------------------------------------------ build model data
idx_target = TARGET
idx_pred = [FIELDS.index(p) for p in predictors]
use = np.isfinite(MATRIX[:n_rows, idx_target])
sub = MATRIX[:n_rows][use][:, [idx_target] + idx_pred]
y_all = sub[:, 0]
X_all = sub[:, 1:]
# median imputation for missing predictors (documented in the thesis write-up)
for j in range(X_all.shape[1]):
    col = X_all[:, j]
    bad = ~np.isfinite(col)
    if bad.any():
        col[bad] = np.nanmedian(col)
    X_all[:, j] = col

rng = np.random.default_rng(42)
perm = rng.permutation(len(y_all))
cut = int(0.8 * len(perm))
tr, te = perm[:cut], perm[cut:]
X_tr, X_te, y_tr, y_te = X_all[tr], X_all[te], y_all[tr], y_all[te]

Xc_tr = design(X_tr)
ols = OLSFit(Xc_tr, y_tr)
names = ["const"] + predictors
ABS_ERR = 0.0

st.sidebar.caption(f"Train n={len(y_tr):,} · Test n={len(y_te):,}")

# ------------------------------------------------------------------- Overview
if page == "📊 Overview":
    st.subheader("Project overview")
    a, b, c, d = st.columns(4)
    a.metric("Observations modelled", f"{len(y_all):,}")
    b.metric("Predictors", len(predictors))
    c.metric("Target", "PM2.5")
    d.metric("Study area", "Bengaluru")

    st.info(
        "**Phase 1** validates a classical OLS estimator (fit, diagnostics, "
        "bootstrap). **Phase 2** compares penalised and non-linear learners and "
        "selects a final model. Every number below is computed live from the "
        "processed data rather than hard-coded."
    )

    st.markdown("#### Descriptive statistics")
    desc = []
    for j, name in enumerate(FIELDS):
        col = MATRIX[:, j]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        desc.append({
            "Variable": LABELS[name], "n": len(col),
            "Mean": round(float(col.mean()), 2),
            "SD": round(float(col.std(ddof=1)), 2),
            "Min": round(float(col.min()), 2),
            "Median": round(float(np.median(col)), 2),
            "Max": round(float(col.max()), 2),
        })
    st.table(desc)

    st.markdown("#### Correlation matrix")
    cols = [TARGET] + idx_pred
    corr = np.corrcoef(MATRIX[:, cols], rowvar=False)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    tick = [LABELS[FIELDS[j]].split(" ")[0] for j in cols]
    ax.set_xticks(range(len(cols)), tick, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), tick)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation between PM2.5 and selected predictors")
    fig_show(fig)

    st.markdown("#### Target distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_all, bins=50, color="#1f77b4", edgecolor="white")
    axes[0].set_xlabel(LABELS["PM2.5"])
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram of PM2.5")
    axes[1].boxplot(y_all, vert=False)
    axes[1].set_xlabel(LABELS["PM2.5"])
    axes[1].set_title("Boxplot (outlier structure)")
    fig_show(fig)

# -------------------------------------------------------- Phase 1 regression
elif page == "🔬 Phase 1 · Classical Regression":
    st.subheader("Phase 1 · Classical regression")

    st.markdown("### Simple linear regression — PM2.5 = β₀ + β₁·PM10 + ε")
    if "PM10" in predictors:
        j = predictors.index("PM10")
        m = OLSFit(design(X_tr[:, [j]]), y_tr)
        slr_pred = m.beta[0] + m.beta[1] * X_te[:, j]
        r2, rmse, mae = metrics(y_te, slr_pred)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("β₀", f"{m.beta[0]:.3f}")
        c2.metric("β₁ (PM10)", f"{m.beta[1]:.4f}")
        c3.metric("R² (test)", f"{r2:.4f}")
        c4.metric("RMSE (test)", f"{rmse:.3f}")
        st.code(f"PM2.5 = {m.beta[0]:.3f} + {m.beta[1]:.4f} × PM10   "
                f"(t = {m.tvalues[1]:.2f}, p = {m.pvalues[1]:.3g})", language="text")
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(X_te[:, j], y_te, s=8, alpha=0.35, label="Observed")
            order = np.argsort(X_te[:, j])
            ax.plot(X_te[order, j], slr_pred[order], "r-", lw=2, label="Fitted")
            ax.set_xlabel(LABELS["PM10"])
            ax.set_ylabel(LABELS["PM2.5"])
            ax.set_title("SLR fit")
            ax.legend()
            fig_show(fig)
        with right:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(m.fitted, m.resid, s=8, alpha=0.35)
            ax.axhline(0, color="red", ls="--")
            ax.set_xlabel("Fitted")
            ax.set_ylabel("Residual")
            ax.set_title("SLR residual plot")
            fig_show(fig)
    else:
        st.warning("Include PM10 to see the simple linear regression.")

    st.markdown("### Multiple linear regression")
    mlr = LinearRegression().fit(X_tr, y_tr)
    pred = mlr.predict(X_te)
    r2, rmse, mae = metrics(y_te, pred)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (train)", f"{mlr.score(X_tr, y_tr):.4f}")
    c2.metric("R² (test)", f"{r2:.4f}")
    c3.metric("RMSE (test)", f"{rmse:.3f}")
    c4.metric("MAE (test)", f"{mae:.3f}")

    st.markdown("#### Fitted equation")
    st.code("PM2.5 = " + f"{ols.beta[0]:.3f}" + "".join(
        f" + ({b:.3f})·{p}" for b, p in zip(ols.beta[1:], predictors)), language="text")

    st.markdown("#### Overall fit")
    st.write(
        f"R² = **{ols.r2:.4f}** · Adjusted R² = **{ols.r2_adj:.4f}** · "
        f"F = **{ols.fvalue:.1f}** (p = {ols.f_pvalue:.3g}) · "
        f"AIC = **{ols.aic:.1f}** · BIC = **{ols.bic:.1f}** · n = {ols.n:,}"
    )

    st.markdown("#### How each component contributes to the prediction")
    means = X_tr.mean(axis=0)
    contrib = ols.beta[1:] * means
    total = np.abs(contrib).sum() + abs(ols.beta[0])
    rows = []
    for i, p in enumerate(predictors):
        rows.append({
            "Predictor": p,
            "β (unstandardised)": round(float(ols.beta[i + 1]), 4),
            "Mean level": round(float(means[i]), 2),
            "Mean contribution β·x̄": round(float(contrib[i]), 3),
            "Share of prediction %": round(float(100 * abs(contrib[i]) / total), 1),
            "t": round(float(ols.tvalues[i + 1]), 2),
            "p-value": f"{ols.pvalues[i + 1]:.3g}",
        })
    st.table(rows)
    st.caption(
        f"Intercept contributes {ols.beta[0]:.2f} µg/m³; the listed terms sum the "
        f"remaining {contrib.sum():.2f} µg/m³ at the mean observation."
    )

    sd_x = X_tr.std(axis=0, ddof=1)
    sd_y = y_tr.std(ddof=1)
    beta_std = ols.beta[1:] * sd_x / sd_y
    order = np.argsort(beta_std)

    g1, g2 = st.columns(2)
    with g1:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        colours = ["#2ca02c" if v >= 0 else "#d62728" for v in beta_std[order]]
        ax.barh([predictors[i] for i in order], beta_std[order], color=colours)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Standardised β (SD change in PM2.5 per 1 SD of X)")
        ax.set_title("Relative importance of predictors")
        fig_show(fig)
    with g2:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        labels = ["Intercept"] + list(predictors)
        values = [ols.beta[0]] + list(contrib)
        colours = ["#7f7f7f"] + ["#1f77b4" if v >= 0 else "#d62728" for v in contrib]
        ax.bar(labels, values, color=colours)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Contribution at mean observation (µg/m³)")
        ax.set_title("Waterfall of component contributions")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig_show(fig)

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_te, pred, s=8, alpha=0.35)
        lo, hi = float(min(y_te.min(), pred.min())), float(max(y_te.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.set_title("Predicted vs actual (test set)")
        fig_show(fig)
    with p2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(ols.fitted, ols.resid, s=8, alpha=0.35)
        ax.axhline(0, color="red", ls="--")
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Residual vs fitted")
        fig_show(fig)

    st.markdown("#### Coefficient table with inference")
    st.table([
        {"Term": n, "β": round(float(b), 4), "SE": round(float(s), 4),
         "t": round(float(t), 2), "p": f"{p:.3g}"}
        for n, b, s, t, p in zip(names, ols.beta, ols.se, ols.tvalues, ols.pvalues)
    ])

# ------------------------------------------------------- Phase 1 assumptions
elif page == "✅ Phase 1 · Assumption Testing":
    st.subheader("Phase 1 · Classical assumption diagnostics")

    vif = vif_values(X_tr)
    bp_lm, bp_p = breusch_pagan(ols, X_tr)
    jb, jb_p, skew, kurt = jarque_bera(ols.resid)
    dw = durbin_watson(ols.resid)
    sh_W, sh_p = stats.shapiro(ols.resid[: min(5000, len(ols.resid))])

    st.markdown("#### Consolidated assumption table")
    st.table([
        {"Assumption": "Linearity", "Test": "Residual vs fitted plot",
         "Result": "see plot", "Interpretation": "Curvature ⇒ non-linear structure"},
        {"Assumption": "Multicollinearity", "Test": "VIF / tolerance",
         "Result": f"max VIF = {vif.max():.2f}",
         "Interpretation": ">10 severe, >5 moderate"},
        {"Assumption": "Homoscedasticity", "Test": "Breusch–Pagan",
         "Result": f"LM = {bp_lm:.2f}, p = {bp_p:.3g}",
         "Interpretation": "p < 0.05 ⇒ variance not constant"},
        {"Assumption": "Normality", "Test": "Jarque–Bera",
         "Result": f"JB = {jb:.1f}, p = {jb_p:.3g}",
         "Interpretation": "p < 0.05 ⇒ residuals non-normal"},
        {"Assumption": "Independence", "Test": "Durbin–Watson",
         "Result": f"DW = {dw:.3f}",
         "Interpretation": "≈2 OK; <1.5 positive autocorrelation"},
    ])

    left, right = st.columns(2)
    with left:
        st.markdown("##### Variance inflation factors")
        st.table([{"Predictor": predictors[j],
                   "VIF": round(float(vif[j]), 2),
                   "Tolerance": round(float(1 / vif[j]), 3),
                   "Verdict": "Severe" if vif[j] > 10 else
                              ("Moderate" if vif[j] > 5 else "OK")}
                  for j in range(len(predictors))])

        st.markdown("##### Heteroskedasticity (Breusch–Pagan)")
        st.write(f"LM statistic = **{bp_lm:.3f}**, p-value = **{bp_p:.4g}**")
        if bp_p < 0.05:
            st.warning("Reject H₀: heteroskedasticity is present. HC3 robust "
                       "standard errors are shown below.")
            rse = ols.robust_se()
            st.table([{"Term": n, "β": round(float(b), 4),
                       "OLS SE": round(float(s), 4),
                       "HC3 SE": round(float(r), 4),
                       "t (robust)": round(float(b / r), 2)}
                      for n, b, s, r in zip(names, ols.beta, ols.se, rse)])
        else:
            st.success("Fail to reject H₀: no evidence of heteroskedasticity.")
    with right:
        st.markdown("##### Residual normality")
        st.write(f"Jarque–Bera = **{jb:.2f}** (p = **{jb_p:.3g}**)")
        st.write(f"Shapiro–Wilk W = **{sh_W:.4f}** (p = **{sh_p:.3g}**, "
                 f"n = {min(5000, len(ols.resid))})")
        st.write(f"Skewness = **{skew:.3f}**, kurtosis = **{kurt:.3f}**")

        st.markdown("##### Residual independence")
        st.write(f"Durbin–Watson = **{dw:.4f}**")
        if 1.5 < dw < 2.5:
            st.success("DW near 2: no strong first-order autocorrelation.")
        else:
            st.warning("DW away from 2: serial dependence is likely — hourly "
                       "pollutant series are autocorrelated, so naive p-values "
                       "understate standard errors.")

    a, b, c = st.columns(3)
    with a:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.scatter(ols.fitted, ols.resid, s=7, alpha=0.3)
        ax.axhline(0, color="red", ls="--")
        ax.set_title("Residual vs fitted")
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        fig_show(fig)
    with b:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        stats.probplot(ols.resid, dist="norm", plot=ax)
        ax.set_title("Normal Q–Q plot")
        fig_show(fig)
    with c:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        ax.scatter(ols.fitted, np.sqrt(np.abs(ols.resid / ols.resid.std())),
                   s=7, alpha=0.3)
        ax.set_title("Scale–location")
        ax.set_xlabel("Fitted")
        ax.set_ylabel("√|standardised residual|")
        fig_show(fig)

# --------------------------------------------------------- Phase 1 bootstrap
elif page == "🎲 Phase 1 · Bootstrap Validation":
    st.subheader("Phase 1 · Bootstrap estimator validation")
    st.caption(
        "Resample the training rows with replacement, refit OLS, and compare the "
        "empirical coefficient distribution with the classical "
        "Var(β̂|X) = σ²(XᵀX)⁻¹ approximation."
    )
    n_boot = st.slider("Bootstrap iterations", 200, 1500, 400, step=100)

    if st.button("Run bootstrap", type="primary"):
        rng_b = np.random.default_rng(7)
        n = len(y_tr)
        store = np.empty((n_boot, X_tr.shape[1]))
        progress = st.progress(0.0)
        for i in range(n_boot):
            rows = rng_b.integers(0, n, n)
            fit = OLSFit(design(X_tr[rows]), y_tr[rows])
            store[i] = fit.beta[1:]
            if i % 25 == 0:
                progress.progress((i + 1) / n_boot)
        progress.progress(1.0)
        st.session_state["boot"] = (store, ols.beta[1:].copy(), ols.se[1:].copy())

    if "boot" in st.session_state:
        store, point, theory_se = st.session_state["boot"]
        ci = np.percentile(store, [2.5, 97.5], axis=0)
        boot_mean = store.mean(axis=0)
        boot_se = store.std(axis=0, ddof=1)
        st.markdown("#### Empirical vs theoretical sampling properties")
        st.table([{
            "Predictor": predictors[j],
            "OLS β̂": round(float(point[j]), 4),
            "Bootstrap mean": round(float(boot_mean[j]), 4),
            "Bootstrap variance": round(float(store[:, j].var(ddof=1)), 6),
            "Theoretical variance": round(float(theory_se[j] ** 2), 6),
            "Bootstrap SE": round(float(boot_se[j]), 5),
            "Theoretical SE": round(float(theory_se[j]), 5),
            "SE ratio": round(float(boot_se[j] / theory_se[j]), 3),
            "95% CI": f"[{ci[0, j]:.4f}, {ci[1, j]:.4f}]",
        } for j in range(len(predictors))])

        k = len(predictors)
        fig, axes = plt.subplots(1, k, figsize=(3.6 * k, 3.6), squeeze=False)
        for j in range(k):
            ax = axes[0][j]
            vals = store[:, j]
            ax.hist(vals, bins=35, density=True, alpha=0.65,
                    color="#1f77b4", edgecolor="white")
            grid = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(grid, stats.norm.pdf(grid, vals.mean(), vals.std(ddof=1)),
                    "k--", lw=1.3, label="normal overlay")
            ax.axvline(vals.mean(), color="red", lw=1.2, label="bootstrap mean")
            ax.axvline(point[j], color="green", ls=":", lw=1.4, label="OLS β̂")
            ax.set_title(predictors[j], fontsize=9)
            ax.legend(fontsize=6)
        fig.suptitle("Empirical sampling distributions of the OLS coefficients")
        fig_show(fig)

        fig, ax = plt.subplots(figsize=(9, 4.4))
        pos = np.arange(k)
        ax.errorbar(pos - 0.12, theory_se, yerr=theory_se, fmt="o",
                    capsize=3, color="#1f77b4", label="Theoretical SE")
        ax.errorbar(pos + 0.12, boot_se, yerr=boot_se, fmt="s",
                    capsize=3, color="#d62728", label="Bootstrap SE")
        ax.set_xticks(pos, predictors, rotation=25, ha="right")
        ax.set_ylabel("Standard error")
        ax.set_title("Theoretical vs empirical standard errors")
        ax.legend()
        fig_show(fig)

        ratio = boot_se / theory_se
        st.markdown("#### Interpretation")
        st.write(
            f"Bootstrap SEs span **{ratio.min():.2f}×–{ratio.max():.2f}×** of the "
            "classical OLS standard errors. Wide gaps indicate violated "
            "homoscedasticity/independence assumptions rather than a faulty "
            "estimator; the bootstrap interval is the more trustworthy one here."
        )
    else:
        st.info("Press **Run bootstrap** to compute the empirical distributions.")

# ---------------------------------------------------- Phase 2 feature select
elif page == "🎯 Phase 2 · Feature Selection":
    st.subheader("Phase 2 · Feature selection")

    st.markdown("### Stepwise selection by AIC")
    f_sel, f_hist = forward_selection(X_tr, y_tr, predictors)
    b_sel, b_hist = backward_elimination(X_tr, y_tr, predictors)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Forward selection**")
        st.table(f_hist) if f_hist else st.write("No improvement from the null model.")
        st.success("Selected: " + (", ".join(f_sel) if f_sel else "none"))
    with c2:
        st.markdown("**Backward elimination**")
        st.table(b_hist) if b_hist else st.write("All predictors retained.")
        st.success("Selected: " + ", ".join(b_sel))

    st.markdown("### Ridge and LASSO on standardised features")
    alpha_r = st.slider("Ridge α", 0.01, 100.0, 1.0, step=0.01)
    alpha_l = st.slider("LASSO α", 0.001, 1.0, 0.01, step=0.001)

    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)
    ols_std = LinearRegression().fit(Xtr_s, y_tr)
    ridge = Ridge(alpha=alpha_r).fit(Xtr_s, y_tr)
    lasso = Lasso(alpha=alpha_l, max_iter=20000).fit(Xtr_s, y_tr)

    st.table([{
        "Predictor": predictors[j],
        "OLS (standardised)": round(float(ols_std.coef_[j]), 4),
        f"Ridge (α={alpha_r})": round(float(ridge.coef_[j]), 4),
        f"LASSO (α={alpha_l})": round(float(lasso.coef_[j]), 4),
    } for j in range(len(predictors))])

    kept = [p for p, c in zip(predictors, lasso.coef_) if abs(c) > 1e-8]
    st.write(f"LASSO shrinks **{len(predictors) - len(kept)}** coefficient(s) to "
             f"zero; retained: {', '.join(kept) if kept else 'none'}.")

    st.markdown("#### Penalised vs unpenalised fit")
    st.table([{
        "Model": name,
        "R² (test)": round(r2_score(y_te, p), 4),
        "RMSE (test)": round(float(np.sqrt(mean_squared_error(y_te, p))), 3),
    } for name, p in [
        ("OLS", ols_std.predict(Xte_s)),
        ("Ridge", ridge.predict(Xte_s)),
        ("LASSO", lasso.predict(Xte_s)),
    ]])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pos = np.arange(len(predictors))
    ax.bar(pos - 0.27, ols_std.coef_, 0.27, label="OLS")
    ax.bar(pos, ridge.coef_, 0.27, label="Ridge")
    ax.bar(pos + 0.27, lasso.coef_, 0.27, label="LASSO")
    ax.set_xticks(pos, predictors, rotation=25, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Standardised coefficient")
    ax.set_title("Coefficient paths: OLS vs Ridge vs LASSO")
    ax.legend()
    fig_show(fig)

# --------------------------------------------------------- Phase 2 ML models
elif page == "🤖 Phase 2 · ML Models":
    st.subheader("Phase 2 · Machine learning regressors")
    st.caption("Same train/test split as Phase 1; SVR is fitted on a 1,500-row "
               "subsample because its kernel matrix grows quadratically.")

    if st.button("Train models", type="primary"):
        results, preds = [], {}
        with st.spinner("Training models..."):
            dt = DecisionTreeRegressor(max_depth=10, min_samples_split=20,
                                       min_samples_leaf=10,
                                       random_state=42).fit(X_tr, y_tr)
            preds["Decision Tree"] = dt.predict(X_te)

            rf = RandomForestRegressor(n_estimators=60, max_depth=12,
                                       random_state=42, n_jobs=-1).fit(X_tr, y_tr)
            preds["Random Forest"] = rf.predict(X_te)

            s = StandardScaler().fit(X_tr)
            sub = min(1500, len(X_tr))
            svr = SVR(kernel="rbf", C=100, epsilon=0.5,
                      gamma="scale").fit(s.transform(X_tr)[:sub], y_tr[:sub])
            preds["SVR (RBF)"] = svr.predict(s.transform(X_te))

            try:
                import xgboost as xgb

                xg = xgb.XGBRegressor(n_estimators=120, max_depth=6,
                                      learning_rate=0.08, subsample=0.9,
                                      colsample_bytree=0.9, random_state=42,
                                      n_jobs=2).fit(X_tr, y_tr)
                preds["XGBoost"] = xg.predict(X_te)
            except Exception as exc:
                st.warning(f"XGBoost unavailable ({type(exc).__name__}); "
                           "continuing without it.")

            for name, p in preds.items():
                r2, rmse, mae = metrics(y_te, p)
                results.append({"Model": name, "R²": round(r2, 4),
                                "RMSE": round(rmse, 3), "MAE": round(mae, 3)})

        st.session_state["ml"] = (results, preds,
                                  rf.feature_importances_.copy())

    if "ml" in st.session_state:
        results, preds, importance = st.session_state["ml"]
        st.table(sorted(results, key=lambda r: -r["R²"]))
        best = max(results, key=lambda r: r["R²"])
        st.success(f"Best test R²: **{best['Model']}** ({best['R²']:.4f})")

        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.bar([r["Model"] for r in results], [r["R²"] for r in results],
               color="#2ca02c")
        ax.set_ylabel("Test R²")
        ax.set_title("Test-set R² by model")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig_show(fig)

        order = np.argsort(importance)
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.barh([predictors[j] for j in order], importance[order], color="#1f77b4")
        ax.set_xlabel("Impurity-based importance")
        ax.set_title("Random Forest feature importance")
        fig_show(fig)
    else:
        st.info("Press **Train models** to fit Decision Tree, Random Forest, "
                "SVR and XGBoost.")

# ------------------------------------------- Phase 2 comparison & selection
elif page == "🏆 Phase 2 · Comparison & Selection":
    st.subheader("Phase 2 · Comparative evaluation and final model")

    with st.spinner("Running 5-fold cross-validation..."):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        Xs = StandardScaler().fit_transform(X_tr)
        specs = [
            ("Linear Regression", LinearRegression(), False),
            ("Ridge", Ridge(alpha=1.0), True),
            ("LASSO", Lasso(alpha=0.01), True),
            ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=42), False),
            ("Random Forest", RandomForestRegressor(n_estimators=40,
                                                   random_state=42, n_jobs=-1), False),
        ]
        cv_rows = []
        for name, model, scaled in specs:
            Xd = Xs if scaled else X_tr
            r2 = cross_val_score(model, Xd, y_tr, cv=kf, scoring="r2")
            rmse = np.sqrt(-cross_val_score(model, Xd, y_tr, cv=kf,
                                            scoring="neg_mean_squared_error"))
            cv_rows.append({"Model": name,
                            "CV R² mean": round(float(r2.mean()), 4),
                            "CV R² sd": round(float(r2.std()), 4),
                            "CV RMSE mean": round(float(rmse.mean()), 3)})

    st.markdown("#### 5-fold cross-validation (training data)")
    st.table(cv_rows)

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar([r["Model"] for r in cv_rows], [r["CV R² mean"] for r in cv_rows],
           yerr=[r["CV R² sd"] for r in cv_rows], capsize=4, color="#1f77b4")
    ax.set_ylabel("Cross-validated R²")
    ax.set_title("Cross-validated performance (mean ± 1 SD)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig_show(fig)

    if "ml" in st.session_state:
        results, preds, _ = st.session_state["ml"]
        st.markdown("#### Held-out test performance")
        st.table(sorted(results, key=lambda r: -r["R²"]))

        best = max(results, key=lambda r: r["R²"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_te, preds[best["Model"]], s=8, alpha=0.35)
        lo, hi = float(y_te.min()), float(y_te.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel("Actual PM2.5")
        ax.set_ylabel("Predicted PM2.5")
        ax.set_title(f"Best model: {best['Model']}")
        fig_show(fig)

        st.markdown("#### Final selection")
        st.info(
            f"**{best['Model']}** gives the best held-out accuracy "
            f"(R² = {best['R²']:.4f}, RMSE = {best['RMSE']:.3f} µg/m³). "
            "Tree ensembles capture non-linear interactions between pollutants "
            "that OLS cannot express, while feature importance keeps the model "
            "interpretable. The OLS model remains the inferential baseline "
            "because its coefficients have direct marginal-effect meaning, and "
            "its diagnostics quantify how far the classical assumptions hold."
        )
    else:
        st.warning("Train the models first in **Phase 2 · ML Models**.")
