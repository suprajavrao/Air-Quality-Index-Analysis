#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import os
from datetime import datetime

os.makedirs('../reports/figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# Load small sample
df = pd.read_pickle('../data/processed/combined_raw_sample.pkl')
df = df.sample(n=10000, random_state=42)  # Use only 10k rows
print(f"Sample shape: {df.shape}")

# Clean columns
df.columns = [col.strip() for col in df.columns]
col_map = {}
for col in df.columns:
    if 'PM2.5' in col: col_map[col] = 'PM2.5'
    elif 'PM10' in col and 'PM2.5' not in col: col_map[col] = 'PM10'
    elif 'NO2' in col: col_map[col] = 'NO2'
    elif col == 'NO' or col.startswith('NO '): col_map[col] = 'NO'
    elif 'SO2' in col: col_map[col] = 'SO2'
    elif 'CO' in col and 'CO2' not in col: col_map[col] = 'CO'
    elif 'Ozone' in col: col_map[col] = 'Ozone'
df = df.rename(columns=col_map)

# EDA
relevant = ['PM2.5','PM10','NO','NO2','SO2','CO','Ozone']
avail = [c for c in relevant if c in df.columns]
dfc = df[avail].copy()
for c in dfc.columns: dfc[c] = pd.to_numeric(dfc[c], errors='coerce')
dfc = dfc.dropna(subset=['PM2.5']).fillna(dfc.median())
print(f"Cleaned: {dfc.shape}")
dfc.describe().to_csv('../results/descriptive_statistics.csv')
corr = dfc.corr()
plt.figure(figsize=(8,6)); sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f'); plt.title('Correlation Matrix'); plt.tight_layout(); plt.savefig('../reports/figures/correlation_matrix.png',dpi=150); plt.close()
corr['PM2.5'].to_csv('../results/correlation_with_pm25.csv')

# Regression
features = [c for c in ['PM10','NO','NO2','SO2','CO','Ozone'] if c in dfc.columns]
X = dfc[features]; y = dfc['PM2.5']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# SLR
lr_slr = LinearRegression().fit(X_train[['PM10']], y_train)
plt.figure(figsize=(8,5)); plt.scatter(X_test[['PM10']], y_test, alpha=0.3); plt.plot(X_test[['PM10']], lr_slr.predict(X_test[['PM10']]), color='red'); plt.xlabel('PM10'); plt.ylabel('PM2.5'); plt.title('SLR'); plt.savefig('../reports/figures/slr_regression_line.png',dpi=150); plt.close()

# MLR
lr_mlr = LinearRegression().fit(X_train,y_train)
yp = lr_mlr.predict(X_test)
print(f"MLR R²: {r2_score(y_test,yp):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test,yp)):.4f}")
for f,c in zip(features,lr_mlr.coef_): print(f"  {f}: {c:.4f}")
Xc = sm.add_constant(X_train); m = sm.OLS(y_train,Xc).fit()
with open('../results/mlr_summary.txt','w') as f: f.write(str(m.summary()))
pd.DataFrame({'Variable':['Intercept']+features,'Coefficient':[lr_mlr.intercept_]+list(lr_mlr.coef_)}).to_csv('../results/mlr_coefficients.csv',index=False)

# Assumptions
resid = m.resid; yp_tr = m.fittedvalues
plt.figure(figsize=(8,5)); plt.scatter(yp_tr,resid,alpha=0.3); plt.axhline(0,color='red'); plt.title('Residual vs Fitted'); plt.savefig('../reports/figures/residual_vs_fitted.png',dpi=150); plt.close()
vif = pd.DataFrame({'Variable':X.columns,'VIF':[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]})
vif.to_csv('../results/vif_results.csv',index=False); print(vif)
try:
    bp = het_breuschpagan(resid, Xc); print(f"BP p-value: {bp[1]:.4f}")
except: pass
print(f"DW: {durbin_watson(resid):.4f}")
plt.figure(figsize=(8,5)); sns.histplot(resid,kde=True); plt.title('Residuals'); plt.savefig('../reports/figures/residual_histogram.png',dpi=150); plt.close()

# ML
sc = StandardScaler(); Xtr=X_train.copy(); Xte=X_test.copy()
Xtrs=sc.fit_transform(Xtr); Xtes=sc.transform(Xte)
res=[]
for name,model in [('Ridge',Ridge(1.0)),('LASSO',Lasso(0.01))]:
    model.fit(Xtrs,y_train); ypr=model.predict(Xtes)
    res.append({'Model':name,'R2':r2_score(y_test,ypr),'RMSE':np.sqrt(mean_squared_error(y_test,ypr))})
rf=RandomForestRegressor(n_estimators=50,random_state=42,n_jobs=-1).fit(Xtr,y_train); ypr=rf.predict(Xte)
res.append({'Model':'Random Forest','R2':r2_score(y_test,ypr),'RMSE':np.sqrt(mean_squared_error(y_test,ypr))})
try:
    xg=xgb.XGBRegressor(n_estimators=50,random_state=42).fit(Xtr,y_train); ypr=xg.predict(Xte)
    res.append({'Model':'XGBoost','R2':r2_score(y_test,ypr),'RMSE':np.sqrt(mean_squared_error(y_test,ypr))})
except: pass
rdf=pd.DataFrame(res); rdf.to_csv('../results/model_comparison.csv',index=False); print(rdf)

# Report
report = f"""STATISTICAL ANALYSIS AND PREDICTIVE MODELLING OF AIR QUALITY IN BENGALURU
USING CLASSICAL REGRESSION AND MACHINE LEARNING TECHNIQUES

Date: {datetime.now()}
Location: Bengaluru, Karnataka, India
Target Variable: PM2.5

Dataset: {dfc.shape[0]} observations

Model Comparison:
{rdf.to_string()}

MLR Coefficients:
{pd.DataFrame({'Variable':features,'Coefficient':lr_mlr.coef_}).to_string()}
"""
with open('../reports/Final_Project_Report.txt','w') as f: f.write(report)
print("\n=== REPORT ===\n", report)
