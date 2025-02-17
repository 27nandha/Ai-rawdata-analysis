import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.gemini_helper import GeminiAnalyzer
from utils.report_generator import ReportGenerator
import base64

def get_ai_insights(df):
    """Get AI insights about data quality issues"""
    analyzer = GeminiAnalyzer(df)
    prompt = """Analyze this dataset for data quality issues. Consider:
    1. Missing values and their patterns
    2. Potential outliers
    3. Data type inconsistencies
    4. Value distributions
    Provide specific recommendations for cleaning this dataset."""
    
    try:
        insights = analyzer.suggest_visualization(df.columns)
        return insights
    except Exception as e:
        return f"Error getting AI insights: {str(e)}"

def clean_dataset(df):
    """Clean the dataset using specified methods"""
    cleaned_df = df.copy()
    
    # Get numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    
    # 1. Handle missing values with KNN Imputer
    if st.checkbox("Handle Missing Values (KNN Imputer)"):
        n_neighbors = st.slider("Select number of neighbors for KNN", 1, 10, 5)
        imputer = KNNImputer(n_neighbors=n_neighbors)
        if len(numeric_cols) > 0:
            cleaned_df[numeric_cols] = imputer.fit_transform(cleaned_df[numeric_cols])
            st.success("Missing values handled using KNN Imputer")
    
    # 2. Detect and remove outliers
    if st.checkbox("Remove Outliers (Isolation Forest)"):
        if len(numeric_cols) > 0:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(cleaned_df[numeric_cols])
            n_outliers = (outlier_labels == -1).sum()
            st.write(f"Found {n_outliers} outliers")
            if st.button("Remove Detected Outliers"):
                cleaned_df = cleaned_df[outlier_labels == 1]
                st.success(f"Removed {n_outliers} outliers")
    
    # 3. Scale numerical features
    if st.checkbox("Scale Numerical Features"):
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            cleaned_df[numeric_cols] = scaler.fit_transform(cleaned_df[numeric_cols])
            st.success("Numerical features scaled")
    
    return cleaned_df

def create_download_link(val, filename):
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}">Download Report</a>'

def main():
    st.title("🧹 Data Cleaning")
    
    # 1. Load Dataset
    st.subheader("1. Load Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:", df.head())
        st.write("Shape:", df.shape)
        
        # 2. Get AI Insights
        st.subheader("2. Get AI Insights")
        if st.button("Analyze Data Quality"):
            with st.spinner("Getting AI insights..."):
                insights = get_ai_insights(df)
                st.write(insights)
        
        # 3. Clean Data
        st.subheader("3. Clean Data")
        st.write("Select cleaning operations:")
        
        cleaned_df = clean_dataset(df)
        
        # 4. Show Results & Save
        if st.checkbox("Show Cleaning Results"):
            st.subheader("4. Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Data Info:")
                st.write(f"- Rows: {df.shape[0]}")
                st.write(f"- Missing Values: {df.isnull().sum().sum()}")
            
            with col2:
                st.write("Cleaned Data Info:")
                st.write(f"- Rows: {cleaned_df.shape[0]}")
                st.write(f"- Missing Values: {cleaned_df.isnull().sum().sum()}")
        
        # 5. Save Cleaned Data
        if st.button("Save Cleaned Data"):
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download Cleaned Dataset",
                data=csv,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )
        
        # 6. Generate Report
        st.subheader("6. Generate Report")
        if st.button("Generate Analysis Report"):
            with st.spinner("Generating comprehensive report..."):
                try:
                    report_gen = ReportGenerator(df, cleaned_df)
                    pdf = report_gen.generate_report()
                    html = create_download_link(pdf, "dataset_analysis_report.pdf")
                    st.markdown(html, unsafe_allow_html=True)
                    st.success("Report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Store cleaned data for other pages
        st.session_state['cleaned_data'] = cleaned_df

if __name__ == "__main__":
    main() 