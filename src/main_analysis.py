#!/usr/bin/env python3
"""
Comprehensive Air Quality Analysis for Bengaluru
Statistical Analysis and Predictive Modelling of Air Quality in Bengaluru
Using Classical Regression and Machine Learning Techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
import xgboost as xgb
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson

import os
from datetime import datetime

# Create directories
os.makedirs('../reports/figures', exist_ok=True)
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

class AirQualityAnalysis:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare data"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        try:
            # Try to load pickle first
            self.df = pd.read_pickle('../data/processed/combined_raw_sample.pkl')
            print(f"Loaded data shape: {self.df.shape}")
        except:
            # Load from combined raw
            self.df = pd.read_csv('../data/processed/combined_raw_sample.csv')
            print(f"Loaded data shape: {self.df.shape}")
        
        # Clean column names
        self.df.columns = [col.strip() for col in self.df.columns]
        # Fix encoding issues in column names
        col_map = {}
        for col in self.df.columns:
            if 'PM2.5' in col:
                col_map[col] = 'PM2.5'
            elif 'PM10' in col and 'PM2.5' not in col:
                col_map[col] = 'PM10'
            elif col.startswith('NO2') or 'NO2' in col:
                col_map[col] = 'NO2'
            elif col.startswith('NO ') or col == 'NO':
                col_map[col] = 'NO'
            elif 'NOx' in col:
                col_map[col] = 'NOx'
            elif 'SO2' in col:
                col_map[col] = 'SO2'
            elif col.startswith('CO') and 'CO2' not in col:
                col_map[col] = 'CO'
            elif 'Ozone' in col:
                col_map[col] = 'Ozone'
            elif 'NH3' in col:
                col_map[col] = 'NH3'
            elif col == 'AT' or 'Temp' in col:
                col_map[col] = 'Temperature'
            elif 'RH' in col:
                col_map[col] = 'RelativeHumidity'
            elif 'WS' in col and 'WD' not in col and 'VWS' not in col:
                col_map[col] = 'WindSpeed'
            elif 'WD' in col:
                col_map[col] = 'WindDirection'
            elif 'BP' in col:
                col_map[col] = 'BP'
        
        self.df = self.df.rename(columns=col_map)
        print(f"Renamed columns. Available: {[c for c in self.df.columns if c in ['PM2.5','PM10','NO2','NO','SO2','CO','Ozone','Temperature','RelativeHumidity','WindSpeed']]}")
        return self.df
    
    def eda(self):
        """Exploratory Data Analysis"""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*60)
        
        # Select relevant variables
        relevant_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'SO2', 'CO', 'Ozone', 
                         'NH3', 'Temperature', 'RelativeHumidity', 'WindSpeed']
        available_cols = [c for c in relevant_cols if c in self.df.columns]
        df_eda = self.df[available_cols].copy()
        
        # Data cleaning - convert to numeric and handle missing
        for col in df_eda.columns:
            df_eda[col] = pd.to_numeric(df_eda[col], errors='coerce')
        
        # Drop rows with missing PM2.5 (target)
        df_eda = df_eda.dropna(subset=['PM2.5'])
        # Fill remaining missing with median
        df_eda = df_eda.fillna(df_eda.median())
        
        self.df_clean = df_eda
        print(f"Cleaned data shape: {self.df_clean.shape}")
        
        # Summary statistics
        print("\n=== Descriptive Statistics ===")
        desc = self.df_clean.describe()
        print(desc)
        desc.to_csv('../results/descriptive_statistics.csv')
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        corr = self.df_clean.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', square=True)
        plt.title('Correlation Matrix of Air Quality Parameters - Bengaluru', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../reports/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved correlation matrix")
        
        # Correlation with PM2.5
        print("\n=== Correlation with PM2.5 ===")
        pm25_corr = corr['PM2.5'].sort_values(ascending=False)
        print(pm25_corr)
        pm25_corr.to_csv('../results/correlation_with_pm25.csv')
        
        # Pairplot
        plt.figure(figsize=(15, 15))
        sns.pairplot(self.df_clean, diag_kind='kde')
        plt.suptitle('Pairplot of Air Quality Variables', y=1.02, fontsize=16)
        plt.savefig('../reports/figures/pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved pairplot")
        
        # Distribution of PM2.5
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df_clean['PM2.5'], kde=True, bins=50)
        plt.title('Distribution of PM2.5 Concentration in Bengaluru', fontsize=14, fontweight='bold')
        plt.xlabel('PM2.5 (µg/m³)')
        plt.ylabel('Frequency')
        plt.savefig('../reports/figures/pm25_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved PM2.5 distribution plot")
        
        return self.df_clean
    
    def simple_linear_regression(self):
        """Simple Linear Regression: PM2.5 vs PM10"""
        print("\n" + "="*60)
        print("SIMPLE LINEAR REGRESSION (SLR)")
        print("="*60)
        print("Model: PM2.5 = β0 + β1(PM10) + ε")
        
        df_slr = self.df_clean[['PM2.5', 'PM10']].dropna()
        X = df_slr[['PM10']]
        y = df_slr['PM2.5']
        
        # Train-test split
        self.X_train_slr, self.X_test_slr, self.y_train_slr, self.y_test_slr = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Fit OLS model
        lr_slr = LinearRegression()
        lr_slr.fit(self.X_train_slr, self.y_train_slr)
        
        # Predictions
        y_pred_train = lr_slr.predict(self.X_train_slr)
        y_pred_test = lr_slr.predict(self.X_test_slr)
        
        # Metrics
        r2_train = r2_score(self.y_train_slr, y_pred_train)
        r2_test = r2_score(self.y_test_slr, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(self.y_train_slr, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(self.y_test_slr, y_pred_test))
        mae_test = mean_absolute_error(self.y_test_slr, y_pred_test)
        
        print(f"\nIntercept (β0): {lr_slr.intercept_:.4f}")
        print(f"Coefficient (β1) for PM10: {lr_slr.coef_[0]:.4f}")
        print(f"R² (Train): {r2_train:.4f}")
        print(f"R² (Test): {r2_test:.4f}")
        print(f"RMSE (Train): {rmse_train:.4f}")
        print(f"RMSE (Test): {rmse_test:.4f}")
        print(f"MAE (Test): {mae_test:.4f}")
        
        # Statistical summary using statsmodels
        X_const = sm.add_constant(self.X_train_slr)
        model_sm = sm.OLS(self.y_train_slr, X_const).fit()
        print("\n=== OLS Summary ===")
        print(model_sm.summary())
        
        # Save summary
        with open('../results/slr_summary.txt', 'w') as f:
            f.write(str(model_sm.summary()))
        
        # Visualizations
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_test_slr, self.y_test_slr, alpha=0.5, label='Actual')
        plt.plot(self.X_test_slr, y_pred_test, color='red', linewidth=2, label='Predicted')
        plt.xlabel('PM10 (µg/m³)')
        plt.ylabel('PM2.5 (µg/m³)')
        plt.title('Simple Linear Regression: PM2.5 vs PM10', fontsize=14, fontweight='bold')
        plt.legend()
        plt.savefig('../reports/figures/slr_regression_line.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residual plot
        residuals = self.y_test_slr - y_pred_test
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred_test, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('SLR Residual Plot', fontsize=14, fontweight='bold')
        plt.savefig('../reports/figures/slr_residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.model_slr = lr_slr
        self.model_slr_sm = model_sm
        return model_sm
    
    def multiple_linear_regression(self):
        """Multiple Linear Regression"""
        print("\n" + "="*60)
        print("MULTIPLE LINEAR REGRESSION (MLR)")
        print("="*60)
        
        # Select independent variables (exclude PM2.5 as target)
        features = ['PM10', 'NO', 'NO2', 'SO2', 'CO', 'Ozone']
        features = [f for f in features if f in self.df_clean.columns]
        print(f"Features used: {features}")
        
        df_mlr = self.df_clean[features + ['PM2.5']].dropna()
        X = df_mlr[features]
        y = df_mlr['PM2.5']
        
        self.X_mlr = X
        self.y_mlr = y
        self.features_mlr = features
        
        # Train-test split
        self.X_train_mlr, self.X_test_mlr, self.y_train_mlr, self.y_test_mlr = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Fit OLS model
        lr_mlr = LinearRegression()
        lr_mlr.fit(self.X_train_mlr, self.y_train_mlr)
        
        # Predictions
        y_pred_train = lr_mlr.predict(self.X_train_mlr)
        y_pred_test = lr_mlr.predict(self.X_test_mlr)
        
        # Metrics
        r2_train = r2_score(self.y_train_mlr, y_pred_train)
        r2_test = r2_score(self.y_test_mlr, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(self.y_train_mlr, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(self.y_test_mlr, y_pred_test))
        mae_test = mean_absolute_error(self.y_test_mlr, y_pred_test)
        mse_test = mean_squared_error(self.y_test_mlr, y_pred_test)
        
        print(f"\nIntercept: {lr_mlr.intercept_:.4f}")
        for feat, coef in zip(features, lr_mlr.coef_):
            print(f"Coefficient for {feat}: {coef:.4f}")
        print(f"\nR² (Train): {r2_train:.4f}")
        print(f"R² (Test): {r2_test:.4f}")
        print(f"RMSE (Train): {rmse_train:.4f}")
        print(f"RMSE (Test): {rmse_test:.4f}")
        print(f"MAE (Test): {mae_test:.4f}")
        print(f"MSE (Test): {mse_test:.4f}")
        
        # Statistical summary
        X_const = sm.add_constant(self.X_train_mlr)
        model_sm_mlr = sm.OLS(self.y_train_mlr, X_const).fit()
        print("\n=== MLR OLS Summary ===")
        print(model_sm_mlr.summary())
        
        with open('../results/mlr_summary.txt', 'w') as f:
            f.write(str(model_sm_mlr.summary()))
        
        # Save coefficients
        coef_df = pd.DataFrame({
            'Variable': ['Intercept'] + features,
            'Coefficient': [lr_mlr.intercept_] + list(lr_mlr.coef_)
        })
        coef_df.to_csv('../results/mlr_coefficients.csv', index=False)
        
        self.model_mlr = lr_mlr
        self.model_mlr_sm = model_sm_mlr
        return model_sm_mlr
    
    def test_assumptions(self):
        """Test Classical Regression Assumptions"""
        print("\n" + "="*60)
        print("CLASSICAL REGRESSION ASSUMPTION TESTING")
        print("="*60)
        
        # Use MLR model
        X_train = self.X_train_mlr
        y_train = self.y_train_mlr
        model = self.model_mlr_sm
        
        # Get predictions and residuals
        y_pred = model.fittedvalues
        residuals = model.resid
        
        results = []
        
        # 1. Linearity - Residual vs Fitted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.title('Residual vs Fitted Plot - Linearity Check', fontsize=14, fontweight='bold')
        plt.savefig('../reports/figures/residual_vs_fitted.png', dpi=300, bbox_inches='tight')
        plt.close()
        results.append(['Linearity', 'Residual vs Fitted Plot', 'Checked', 'Visual inspection required'])
        print("1. Linearity: Residual vs Fitted plot saved")
        
        # 2. Multicollinearity - VIF
        print("\n2. MULTICOLLINEARITY (VIF)")
        X_vif = self.X_mlr.copy()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_vif.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        print(vif_data)
        vif_data.to_csv('../results/vif_results.csv', index=False)
        results.append(['Multicollinearity', 'VIF', str(vif_data.to_dict()), 'VIF > 10 indicates high multicollinearity'])
        
        # Correlation matrix for multicollinearity
        plt.figure(figsize=(10, 8))
        corr_mlr = X_vif.corr()
        sns.heatmap(corr_mlr, annot=True, cmap='coolwarm', fmt='.3f')
        plt.title('Correlation Matrix - Multicollinearity Check', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../reports/figures/mlr_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heteroskedasticity - Breusch-Pagan
        print("\n3. HETEROSKEDASTICITY")
        try:
            bp_test = het_breuschpagan(residuals, X_const)
            bp_labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
            print(dict(zip(bp_labels, bp_test)))
            results.append(['Homoscedasticity', 'Breusch-Pagan', f"p-value={bp_test[1]:.4f}", 'H0: Homoskedasticity (p>0.05)'])
        except Exception as e:
            print(f"Error in Breusch-Pagan: {e}")
            results.append(['Homoscedasticity', 'Breusch-Pagan', 'Error', str(e)])
        
        # 4. Normality of residuals
        print("\n4. RESIDUAL NORMALITY")
        # Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=50)
        plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
        plt.xlabel('Residuals')
        plt.savefig('../reports/figures/residual_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Q-Q plot
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals', fontsize=14, fontweight='bold')
        plt.savefig('../reports/figures/qq_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Shapiro-Wilk (on sample if large)
        if len(residuals) > 5000:
            sample_resid = np.random.choice(residuals, 5000, replace=False)
        else:
            sample_resid = residuals
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample_resid)
            print(f"Shapiro-Wilk Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
        except:
            shapiro_p = np.nan
        
        # Jarque-Bera
        try:
            jb_stat, jb_p = stats.jarque_bera(residuals)
            print(f"Jarque-Bera Test: Statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
            results.append(['Normality', 'Jarque-Bera', f"p-value={jb_p:.4f}", 'H0: Normal distribution (p>0.05)'])
        except Exception as e:
            print(f"Error in Jarque-Bera: {e}")
            results.append(['Normality', 'Jarque-Bera', 'Error', str(e)])
        
        # 5. Independence - Durbin-Watson
        print("\n5. RESIDUAL INDEPENDENCE (Durbin-Watson)")
        dw = durbin_watson(residuals)
        print(f"Durbin-Watson Statistic: {dw:.4f}")
        print("(Values near 2 indicate no autocorrelation. 1.5 < DW < 2.5 generally acceptable)")
        results.append(['Independence', 'Durbin-Watson', f'{dw:.4f}', 'DW ~ 2 indicates no autocorrelation'])
        
        # Save assumption test results
        results_df = pd.DataFrame(results, columns=['Assumption', 'Test', 'Result', 'Interpretation'])
        results_df.to_csv('../results/assumption_test_results.csv', index=False)
        print("\n=== Assumption Test Summary ===")
        print(results_df)
        
        return results_df
    
    def bootstrap_validation(self, n_bootstrap=1000):
        """Bootstrap Estimator Validation"""
        print("\n" + "="*60)
        print("BOOTSTRAP ESTIMATOR VALIDATION")
        print("="*60)
        print(f"Performing {n_bootstrap} bootstrap iterations...")
        
        X = self.X_train_mlr.copy()
        y = self.y_train_mlr.copy()
        features = self.features_mlr
        n = len(X)
        
        boot_coefs = {feat: [] for feat in features}
        boot_coefs['Intercept'] = []
        
        for i in range(n_bootstrap):
            if (i + 1) % 200 == 0:
                print(f"  Iteration {i+1}/{n_bootstrap}")
            # Bootstrap sample
            idx = np.random.choice(n, n, replace=True)
            X_boot = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
            y_boot = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
            
            # Fit model
            lr_boot = LinearRegression()
            lr_boot.fit(X_boot, y_boot)
            boot_coefs['Intercept'].append(lr_boot.intercept_)
            for j, feat in enumerate(features):
                boot_coefs[feat].append(lr_boot.coef_[j])
        
        # Calculate bootstrap statistics
        boot_results = []
        print("\n=== Bootstrap Results ===")
        for param in ['Intercept'] + features:
            coefs = np.array(boot_coefs[param])
            boot_mean = np.mean(coefs)
            boot_std = np.std(coefs)
            boot_se = boot_std
            ci_lower = np.percentile(coefs, 2.5)
            ci_upper = np.percentile(coefs, 97.5)
            print(f"{param}: Mean={boot_mean:.4f}, SE={boot_se:.4f}, 95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
            boot_results.append({
                'Parameter': param,
                'Bootstrap_Mean': boot_mean,
                'Bootstrap_SE': boot_se,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper
            })
        
        boot_df = pd.DataFrame(boot_results)
        boot_df.to_csv('../results/bootstrap_results.csv', index=False)
        
        # Visualizations
        for param in features[:3]:  # Plot first 3 features
            plt.figure(figsize=(10, 6))
            coefs = boot_coefs[param]
            plt.hist(coefs, bins=50, density=True, alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(coefs), color='red', linestyle='--', label=f'Mean={np.mean(coefs):.4f}')
            plt.axvline(np.percentile(coefs, 2.5), color='green', linestyle=':', label='95% CI')
            plt.axvline(np.percentile(coefs, 97.5), color='green', linestyle=':')
            plt.title(f'Bootstrap Distribution of {param} Coefficient', fontsize=14, fontweight='bold')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Density')
            plt.legend()
            plt.savefig(f'../reports/figures/bootstrap_{param}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("\nBootstrap validation completed. Results saved.")
        self.boot_coefs = boot_coefs
        return boot_df
    
    def machine_learning_models(self):
        """Train various Machine Learning models"""
        print("\n" + "="*60)
        print("MACHINE LEARNING REGRESSION MODELS")
        print("="*60)
        
        X_train = self.X_train_mlr
        X_test = self.X_test_mlr
        y_train = self.y_train_mlr
        y_test = self.y_test_mlr
        
        # Standardize for some models
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = []
        
        # 1. Ridge Regression
        print("\n1. Ridge Regression")
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        y_pred_ridge = ridge.predict(X_test_scaled)
        results.append(self._evaluate_model('Ridge Regression', y_test, y_pred_ridge))
        print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        
        # 2. LASSO Regression
        print("\n2. LASSO Regression")
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train_scaled, y_train)
        y_pred_lasso = lasso.predict(X_test_scaled)
        results.append(self._evaluate_model('LASSO Regression', y_test, y_pred_lasso))
        print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        print(f"   Non-zero coefficients: {sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")
        
        # 3. Decision Tree
        print("\n3. Decision Tree Regression")
        dt = DecisionTreeRegressor(max_depth=10, random_state=42)
        dt.fit(X_train, y_train)
        y_pred_dt = dt.predict(X_test)
        results.append(self._evaluate_model('Decision Tree', y_test, y_pred_dt))
        print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        
        # 4. Random Forest
        print("\n4. Random Forest Regression")
        rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        results.append(self._evaluate_model('Random Forest', y_test, y_pred_rf))
        print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        
        # Feature importance from RF
        feat_imp = pd.DataFrame({
            'Feature': self.features_mlr,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        feat_imp.to_csv('../results/feature_importance_rf.csv', index=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_imp, x='Importance', y='Feature')
        plt.title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('../reports/figures/feature_importance_rf.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. SVR
        print("\n5. Support Vector Regression")
        svr = SVR(kernel='rbf', C=100, gamma='scale')
        svr.fit(X_train_scaled[:10000], y_train[:10000])  # Use subset due to memory
        y_pred_svr = svr.predict(X_test_scaled[:10000])
        results.append(self._evaluate_model('SVR (RBF)', y_test[:10000], y_pred_svr))
        print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        
        # 6. XGBoost
        print("\n6. XGBoost Regression")
        try:
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            results.append(self._evaluate_model('XGBoost', y_test, y_pred_xgb))
            print(f"   R² = {results[-1]['R2']:.4f}, RMSE = {results[-1]['RMSE']:.4f}")
        except Exception as e:
            print(f"   XGBoost error: {e}")
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('../results/model_comparison.csv', index=False)
        print("\n=== Model Comparison ===")
        print(results_df.to_string(index=False))
        
        self.models_results = results_df
        return results_df
    
    def _evaluate_model(self, name, y_true, y_pred):
        return {
            'Model': name,
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred)
        }
    
    def cross_validation(self):
        """K-Fold Cross Validation"""
        print("\n" + "="*60)
        print("K-FOLD CROSS VALIDATION (k=10)")
        print("="*60)
        
        X = self.X_mlr
        y = self.y_mlr
        X_scaled = self.scaler.fit_transform(X)
        
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'LASSO': Lasso(alpha=0.01),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        }
        
        cv_results = []
        for name, model in models.items():
            print(f"\n{name}:")
            if name in ['Ridge', 'LASSO']:
                scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
                rmse_scores = np.sqrt(-cross_val_score(model, X_scaled, y, cv=kfold, scoring='neg_mean_squared_error'))
            else:
                scores = cross_val_score(model, X, y, cv=kfold, scoring='r2')
                rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error'))
            print(f"  R²: {scores.mean():.4f} (+/- {scores.std():.4f})")
            print(f"  RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
            cv_results.append({
                'Model': name,
                'CV_R2_Mean': scores.mean(),
                'CV_R2_Std': scores.std(),
                'CV_RMSE_Mean': rmse_scores.mean(),
                'CV_RMSE_Std': rmse_scores.std()
            })
        
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv('../results/cross_validation_results.csv', index=False)
        print("\nCross-validation completed.")
        return cv_df
    
    def generate_report(self):
        """Generate final summary report"""
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        report = []
        report.append("="*80)
        report.append("STATISTICAL ANALYSIS AND PREDICTIVE MODELLING OF AIR QUALITY IN BENGALURU")
        report.append("USING CLASSICAL REGRESSION AND MACHINE LEARNING TECHNIQUES")
        report.append("="*80)
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n" + "="*80)
        report.append("\n1. PROJECT OVERVIEW")
        report.append("-"*40)
        report.append("Primary Location: Bengaluru, Karnataka, India")
        report.append("Target Variable: PM2.5 (µg/m³)")
        report.append("Objective: Predict PM2.5 concentration using regression and ML techniques")
        report.append("\n" + "="*80)
        report.append("\n2. DATA SUMMARY")
        report.append("-"*40)
        if hasattr(self, 'df_clean'):
            report.append(f"Dataset Size: {self.df_clean.shape[0]} observations, {self.df_clean.shape[1]} variables")
            report.append(f"\nVariables: {', '.join(self.df_clean.columns)}")
        report.append("\n" + "="*80)
        report.append("\n3. KEY FINDINGS")
        report.append("-"*40)
        if hasattr(self, 'models_results'):
            report.append("\nModel Performance Comparison:")
            report.append(self.models_results.to_string(index=False))
        report.append("\n" + "="*80)
        report.append("\nReport saved successfully.")
        
        report_text = '\n'.join(report)
        with open('../reports/Final_Project_Report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print("\n\nFinal report saved to ../reports/Final_Project_Report.txt")
        return report_text


def main():
    print("Starting Comprehensive Air Quality Analysis...")
    ana = AirQualityAnalysis()
    ana.load_data()
    ana.eda()
    ana.simple_linear_regression()
    ana.multiple_linear_regression()
    ana.test_assumptions()
    ana.bootstrap_validation(n_bootstrap=500)  # Reduced for memory
    ana.machine_learning_models()
    ana.cross_validation()
    ana.generate_report()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
