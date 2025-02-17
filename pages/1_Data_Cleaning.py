import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from utils.gemini_helper import GeminiAnalyzer
from fpdf import FPDF
import plotly.express as px
import io
from datetime import datetime
import os

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

def create_analysis_pdf(df, cleaned_df, insights, visualizations=None):
    """Create a PDF report with data analysis and visualizations"""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Data Analysis Report', ln=True, align='C')
    pdf.ln(10)
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    pdf.ln(10)
    
    # Dataset Overview
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Dataset Overview', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, f"""
    Original Shape: {df.shape}
    Cleaned Shape: {cleaned_df.shape}
    Total Missing Values (Original): {df.isnull().sum().sum()}
    Total Missing Values (Cleaned): {cleaned_df.isnull().sum().sum()}
    """)
    pdf.ln(5)
    
    # Data Quality Insights
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Data Quality Insights', ln=True)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, str(insights))
    pdf.ln(5)
    
    # Basic Statistics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Statistical Summary', ln=True)
    pdf.set_font('Arial', '', 12)
    stats_text = cleaned_df.describe().to_string()
    pdf.multi_cell(0, 10, stats_text)
    pdf.ln(5)
    
    # Visualizations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '4. Data Visualizations', ln=True)
    
    # Create and save visualizations
    numeric_cols = cleaned_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        # Distribution plot
        for col in numeric_cols[:3]:  # First 3 numeric columns
            fig = px.histogram(cleaned_df, x=col, title=f'Distribution of {col}')
            img_path = f'temp_{col}.png'
            fig.write_image(img_path)
            pdf.image(img_path, x=10, w=190)
            os.remove(img_path)  # Clean up
    
    # Recommendations
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '5. Recommendations & Future Improvements', ln=True)
    pdf.set_font('Arial', '', 12)
    recommendations = """
    1. Data Collection Improvements:
       - Implement data validation at entry points
       - Establish consistent data formats
       - Add data quality checks
    
    2. Feature Engineering:
       - Create derived features
       - Implement domain-specific transformations
       - Consider dimensionality reduction
    
    3. Model Development:
       - Test different algorithms
       - Implement cross-validation
       - Regular model retraining
    
    4. Monitoring:
       - Set up data drift detection
       - Monitor model performance
       - Implement automated alerts
    """
    pdf.multi_cell(0, 10, recommendations)
    
    return pdf

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
        
        # 5. Save Cleaned Data and Generate Report
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Save Cleaned Data"):
                csv = cleaned_df.to_csv(index=False)
                st.download_button(
                    label="Download Cleaned Dataset",
                    data=csv,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Generate Analysis Report"):
                with st.spinner("Generating PDF report..."):
                    # Get insights if not already generated
                    if 'insights' not in st.session_state:
                        insights = get_ai_insights(df)
                    else:
                        insights = st.session_state.insights
                    
                    # Create PDF
                    pdf = create_analysis_pdf(df, cleaned_df, insights)
                    
                    # Save PDF to bytes
                    pdf_bytes = pdf.output(dest='S').encode('latin-1')
                    
                    # Download button for PDF
                    st.download_button(
                        label="Download Analysis Report",
                        data=pdf_bytes,
                        file_name="data_analysis_report.pdf",
                        mime="application/pdf"
                    )
        
        # Store cleaned data for other pages
        st.session_state['cleaned_data'] = cleaned_df

if __name__ == "__main__":
    main() 