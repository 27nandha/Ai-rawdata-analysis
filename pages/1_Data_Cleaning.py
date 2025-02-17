import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px

def advanced_clean_data(df):
    """Advanced data cleaning operations"""
    cleaned_df = df.copy()
    
    st.subheader("🔍 Data Quality Report")
    
    # Basic statistics
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_values = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", total_rows)
    col2.metric("Total Columns", total_cols)
    col3.metric("Missing Values", missing_values)
    col4.metric("Duplicates", duplicates)
    
    st.subheader("🧹 Cleaning Operations")
    
    # 1. Handle Missing Values
    if st.checkbox("Handle Missing Values"):
        missing_strategy = st.selectbox(
            "Choose strategy for missing values:",
            ["Simple Imputation", "KNN Imputation", "Drop rows", "Fill with custom value"]
        )
        
        if missing_strategy == "Simple Imputation":
            method = st.selectbox(
                "Choose imputation method:",
                ["mean", "median", "mode", "forward fill", "backward fill"]
            )
            
            for column in cleaned_df.columns:
                if cleaned_df[column].isnull().any():
                    if method in ["mean", "median"] and np.issubdtype(cleaned_df[column].dtype, np.number):
                        if method == "mean":
                            cleaned_df[column].fillna(cleaned_df[column].mean(), inplace=True)
                        else:
                            cleaned_df[column].fillna(cleaned_df[column].median(), inplace=True)
                    elif method == "mode":
                        cleaned_df[column].fillna(cleaned_df[column].mode()[0], inplace=True)
                    elif method == "forward fill":
                        cleaned_df[column].fillna(method='ffill', inplace=True)
                    elif method == "backward fill":
                        cleaned_df[column].fillna(method='bfill', inplace=True)
        
        elif missing_strategy == "KNN Imputation":
            n_neighbors = st.slider("Select number of neighbors", 1, 10, 5)
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=n_neighbors)
                cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
                st.success("KNN Imputation completed for numeric columns!")
            
        elif missing_strategy == "Drop rows":
            cleaned_df = cleaned_df.dropna()
            
        elif missing_strategy == "Fill with custom value":
            custom_value = st.text_input("Enter custom value:")
            if custom_value:
                cleaned_df = cleaned_df.fillna(custom_value)
    
    # 2. Handle Outliers
    if st.checkbox("Handle Outliers"):
        outlier_method = st.selectbox(
            "Choose outlier detection method:",
            ["IQR Method", "Isolation Forest", "Z-Score"]
        )
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if outlier_method == "IQR Method":
            for column in numeric_cols:
                Q1 = cleaned_df[column].quantile(0.25)
                Q3 = cleaned_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                cleaned_df[column] = cleaned_df[column].clip(lower_bound, upper_bound)
                
        elif outlier_method == "Isolation Forest":
            contamination = st.slider("Select contamination factor", 0.01, 0.1, 0.05)
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            outliers = iso_forest.fit_predict(cleaned_df[numeric_cols])
            cleaned_df = cleaned_df[outliers == 1]
            
        elif outlier_method == "Z-Score":
            threshold = st.slider("Select Z-score threshold", 2.0, 4.0, 3.0)
            for column in numeric_cols:
                z_scores = np.abs((cleaned_df[column] - cleaned_df[column].mean()) / cleaned_df[column].std())
                cleaned_df = cleaned_df[z_scores < threshold]
    
    # 3. Feature Scaling
    if st.checkbox("Apply Feature Scaling"):
        scaling_method = st.selectbox(
            "Choose scaling method:",
            ["StandardScaler", "MinMaxScaler"]
        )
        
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        if scaling_method == "StandardScaler":
            scaler = StandardScaler()
            cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
            
        elif scaling_method == "MinMaxScaler":
            scaler = MinMaxScaler()
            cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
    
    # 4. Remove Duplicates
    if st.checkbox("Remove Duplicate Rows"):
        cleaned_df = cleaned_df.drop_duplicates()
    
    # 5. Fix Data Types
    if st.checkbox("Fix Data Types"):
        for column in cleaned_df.columns:
            current_type = cleaned_df[column].dtype
            new_type = st.selectbox(
                f"Select type for {column} (current: {current_type})",
                ["int64", "float64", "string", "datetime64[ns]", "category", "Keep current"],
                index=5
            )
            if new_type != "Keep current":
                try:
                    if new_type == "datetime64[ns]":
                        cleaned_df[column] = pd.to_datetime(cleaned_df[column])
                    else:
                        cleaned_df[column] = cleaned_df[column].astype(new_type)
                except Exception as e:
                    st.error(f"Error converting {column}: {str(e)}")
    
    return cleaned_df

def main():
    st.title("🧹 Advanced Data Cleaning")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Show original data
        st.subheader("📊 Original Data Preview")
        st.dataframe(df.head())
        
        # Show data info
        st.subheader("ℹ️ Data Information")
        
        # Display missing values visualization
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("Missing Values Distribution:")
            fig = px.bar(x=missing_data.index, y=missing_data.values,
                        labels={'x': 'Columns', 'y': 'Missing Values Count'},
                        title='Missing Values by Column')
            st.plotly_chart(fig)
        
        # Clean data
        cleaned_df = advanced_clean_data(df)
        
        # Show cleaned data
        st.subheader("✨ Cleaned Data Preview")
        st.dataframe(cleaned_df.head())
        
        # Show cleaning impact
        st.subheader("📈 Cleaning Impact")
        col1, col2 = st.columns(2)
        col1.metric("Original Rows", len(df), f"{len(cleaned_df) - len(df)}")
        col2.metric("Missing Values Before", df.isnull().sum().sum(),
                   f"{cleaned_df.isnull().sum().sum() - df.isnull().sum().sum()}")
        
        # Download cleaned data
        if st.button("Download Cleaned Data"):
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        # Store cleaned data in session state
        st.session_state['cleaned_data'] = cleaned_df

if __name__ == "__main__":
    main() 