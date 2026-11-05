#!/usr/bin/env python3
"""Quick version of analysis"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

import os
from datetime import datetime

os.makedirs('../reports/figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

class AirQualityAnalysis:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        self.df = pd.read_pickle('../data/processed/combined_raw_sample.pkl')
        print(f"Loaded data shape: {self.df.shape}")
        self.df.columns = [col.strip() for col in self.df.columns]
        col_map = {}
        for col in self.df.columns:
            if 'PM2.5' in col: col_map[col] = 'PM2.5'
            elif 'PM10' in col and 'PM2.5' not in col: col_map[col] = 'PM10'
            elif col.startswith('NO2') or 'NO2' in col: col_map[col] = 'NO2'
            elif col.startswith('NO ') or col == 'NO': col_map[col] = 'NO'
            elif 'NOx' in col: col_map[col] = 'NOx'
            elif 'SO2' in col: col_map[col] = 'SO2'
            elif col.startswith('CO') and 'CO2' not in col: col_map[col] = 'CO'
            elif 'Ozone' in col: col_map[col] = 'Ozone'
            elif 'NH3' in col: col_map[col] = 'NH3'
            elif col == 'AT' or 'Temp' in col: col_map[col] = 'Temperature'
            elif 'RH' in col: col_map[col] = 'RelativeHumidity'
            elif 'WS' in col and 'WD' not in col and 'VWS' not in col: col_map[col] = 'WindSpeed'
        self.df = self.df.rename(columns=col_map)
        return self.df
    
    def eda(self):
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        relevant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'SO2', 'CO', 'Ozone', 'NH3', 'Temperature', 'RelativeHumidity', 'WindSpeed']
        available_cols = [c for c in relevant_cols if c in self.df.columns]
        df_eda = self.df[available_cols].copy()
        for col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        df_eda = df_eda.dropna(subset=['PM2.5'])
        df_eda = df_eda.fillna(df_eda.median())
        self.df_clean = df_eda
        print(f"Cleaned data shape: {self.df_clean.shape}")
        desc = self.df_clean.describe()
        print(desc)
        desc.to_csv('../results/descriptive_statistics.csv')
        corr = self.df_clean.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../reports/figures/correlation_matrix.png', dpi=150)
        plt.close()
        corr['PM2.5'].to_csv('../results/correlation_with_pm25.csv')
        return self.df_clean
    
    def regression_analysis(self):
        print("\n" + "="*60)
        print("REGRESSION ANALYSIS")
        print("="*60)
        features = ['PM10', 'NO', 'NO2', 'SO2', 'CO', 'Ozone']
        features = [f for f in features if f in self.df_clean.columns]
        df_mlr = self.df_clean[features + ['PM2.5']].dropna()
        X = df_mlr[features]
        y = df_mlr['PM2.5']
        self.X_mlr = X; self.y_mlr = y; self.features_mlr = features
        self.X_train_mlr, self.X_test_mlr, self.y_train_mlr, self.y_test_mlr = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # SLR
        print("\n--- SLR: PM2.5 vs PM10 ---")
        lr_slr = LinearRegression()
        lr_slr.fit(self.X_train_mlr[['PM10']], self.y_train_mlr)
        print(f"R²: {lr_slr.score(self.X_test_mlr[['PM10']], self.y_test_mlr):.4f}, Coef: {lr_slr.coef_[0]:.4f}")
        
        # MLR
        print("\n--- MLR ---")
        lr_mlr = LinearRegression()
        lr_mlr.fit(self.X_train_mlr, self.y_train_mlr)
        y_pred = lr_mlr.predict(self.X_test_mlr)
        print(f"R²: {r2_score(self.y_test_mlr, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(self.y_test_mlr, y_pred)):.4f}")
        for f,c in zip(features, lr_mlr.coef_): print(f"  {f}: {c:.4f}")
        X_const = sm.add_constant(self.X_train_mlr)
        model_sm = sm.OLS(self.y_train_mlr, X_const).fit()
        with open('../results/mlr_summary.txt','w') as f: f.write(str(model_sm.summary()))
        self.model_mlr = lr_mlr; self.model_mlr_sm = model_sm
    
    def assumptions(self):
        print("\n" + "="*60)
        print("ASSUMPTION TESTING")
        print("="*60)
        model = self.model_mlr_sm
        residuals = model.resid
        X_const = sm.add_constant(self.X_train_mlr)
        # VIF
        vif_data = pd.DataFrame({'Variable': self.X_mlr.columns, 'VIF': [variance_inflation_factor(self.X_mlr.values,i) for i in range(self.X_mlr.shape[1])]})
        print(vif_data); vif_data.to_csv('../results/vif_results.csv',index=False)
        # BP
        try:
            bp = het_breuschpagan(residuals, X_const)
            print(f"Breusch-Pagan p-value: {bp[1]:.4f}")
        except: pass
        # DW
        print(f"Durbin-Watson: {durbin_watson(residuals):.4f}")
        plt.figure(figsize=(8,5)); plt.scatter(model.fittedvalues, residuals, alpha=0.3); plt.axhline(0,color='red'); plt.title('Residual vs Fitted'); plt.savefig('../reports/figures/residual_vs_fitted.png',dpi=150); plt.close()
    
    def ml_models(self):
        print("\n" + "="*60)
        print("ML MODELS")
        print("="*60)
        X_train,X_test,y_train,y_test = self.X_train_mlr,self.X_test_mlr,self.y_train_mlr,self.y_test_mlr
        X_tr_s = self.scaler.fit_transform(X_train); X_te_s = self.scaler.transform(X_test)
        res = []
        for name,model in [('Ridge',Ridge(1.0)),('LASSO',Lasso(0.01))]:
            model.fit(X_tr_s,y_train); yp=model.predict(X_te_s)
            res.append({'Model':name,'R2':r2_score(y_test,yp),'RMSE':np.sqrt(mean_squared_error(y_test,yp))})
        rf = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42,n_jobs=-1)
        rf.fit(X_train,y_train); yp=rf.predict(X_test)
        res.append({'Model':'Random Forest','R2':r2_score(y_test,yp),'RMSE':np.sqrt(mean_squared_error(y_test,yp))})
        try:
            xgbm = xgb.XGBRegressor(n_estimators=50,max_depth=5,random_state=42)
            xgbm.fit(X_train,y_train); yp=xgbm.predict(X_test)
            res.append({'Model':'XGBoost','R2':r2_score(y_test,yp),'RMSE':np.sqrt(mean_squared_error(y_test,yp))})
        except: pass
        rdf = pd.DataFrame(res); print(rdf); rdf.to_csv('../results/model_comparison.csv',index=False)
        self.models_results = rdf
    
    def report(self):
        report = f"=== BENGALURU AIR QUALITY ANALYSIS REPORT ===\nDate: {datetime.now()}\n"
        if hasattr(self,'models_results'): report += self.models_results.to_string()
        with open('../reports/Final_Project_Report.txt','w') as f: f.write(report)
        print(report)

def main():
    a = AirQualityAnalysis()
    a.load_data()
    a.eda()
    a.regression_analysis()
    a.assumptions()
    a.ml_models()
    a.report()
    print("\nDONE!")

if __name__=='__main__': main()
