import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bengaluru Air Quality Analysis", layout="wide")

st.title("🌬️ Bengaluru Air Quality Analysis Dashboard")
st.markdown("### Statistical Analysis & Predictive Modelling - Phase 1 & Phase 2")

# Load pre-computed results
try:
    desc = pd.read_csv('results/descriptive_statistics.csv')
    corr = pd.read_csv('results/correlation_with_pm25.csv')
    ml_comp = pd.read_csv('results/model_comparison.csv')
    
    tab1, tab2, tab3 = st.tabs(["📊 Overview & EDA", "🔬 Phase 1 Results", "🤖 Phase 2 Results"])
    
    with tab1:
        st.subheader("Descriptive Statistics")
        st.dataframe(desc)
        st.subheader("Correlation with PM2.5")
        st.dataframe(corr)
        st.image('reports/figures/correlation_matrix.png', use_column_width=True)
        
    with tab2:
        st.subheader("Phase 1: Classical Regression")
        try:
            vif = pd.read_csv('results/vif_results.csv')
            st.write("**Multicollinearity (VIF)**")
            st.dataframe(vif)
        except:
            pass
        try:
            with open('results/mlr_summary.txt') as f:
                st.text(f.read())
        except:
            pass
        st.image('reports/figures/residual_vs_fitted.png', use_column_width=True)
        
    with tab3:
        st.subheader("Phase 2: Model Comparison")
        st.dataframe(ml_comp)
        st.image('reports/figures/feature_importance_rf.png', use_column_width=True) if st.checkbox("Show Feature Importance") else None
        
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Run the analysis script first to generate results.")
