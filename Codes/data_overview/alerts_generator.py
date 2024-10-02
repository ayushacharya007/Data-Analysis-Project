import numpy as np
import streamlit as st

@st.cache_data
def generate_alerts(data):
    '''
    Generate alerts for the given DataFrame.
    
    Args:
        - data: DataFrame to generate alerts for.
        
    Returns:
        - alerts: String containing alerts for the DataFrame.
    '''

    try:
        alerts = ""
        
        if data is not None:
            # 1. Check for high correlation
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr().abs()
            high_corr_pairs = [
                (i, j) for i in corr_matrix.columns for j in corr_matrix.columns 
                if i != j and corr_matrix.loc[i, j] > 0.8
            ]
            for i, j in high_corr_pairs:
                alerts += (
                    f"<p class='alert high-corr-alert'>{i} is highly correlated with {j}"
                    f"<span>High correlation</span></p>"
                )

            # 2. Check for imbalance
            for col in data.columns:
                imbalance_ratio = data[col].value_counts(normalize=True).max()
                if imbalance_ratio > 0.88:
                    alerts += (
                        f"<p class='alert imbalance-alert'>{col} is highly imbalanced "
                        f"({imbalance_ratio * 100:.1f}%)<span>Imbalanced</span></p>"
                    )

            # 3. Check for missing values
            missing_percent = data.isnull().mean() * 100
            for col, percent in missing_percent.items():
                if percent > 45:
                    alerts += (
                        f"<p class='alert missing-alert'>{col} has {percent:.1f}% missing values"
                        f"<span>Missing</span></p>"
                    )

            # 4. Check for unique values
            for col in data.columns:
                if data[col].nunique() / len(data) > 0.6:
                    alerts += (
                        f"<p class='alert unique-alert'>{col} has high unique values"
                        f"<span>Unique</span></p>"
                    )
        
        return alerts

    except Exception as e:
        st.info(f":warning: An error occurred: {e}")