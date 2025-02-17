import pandas as pd
from fpdf import FPDF
import plotly.io as pio
import plotly.express as px
import os
import tempfile

class ReportGenerator:
    def __init__(self, original_df, cleaned_df):
        self.original_df = original_df
        self.cleaned_df = cleaned_df
        
    def _generate_summary(self):
        """Generate a comprehensive summary of the dataset"""
        summary = []
        
        # Dataset Overview Summary
        total_records = len(self.cleaned_df)
        total_features = len(self.cleaned_df.columns)
        numeric_cols = self.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.cleaned_df.select_dtypes(include=['object', 'category']).columns
        
        # Data Quality Summary
        missing_values = self.cleaned_df.isnull().sum().sum()
        missing_percentage = (missing_values / (total_records * total_features)) * 100
        
        # Key Insights
        insights = [
            f"Dataset contains {total_records} records with {total_features} features",
            f"Contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features",
            f"Data completeness: {100 - missing_percentage:.2f}% (Missing: {missing_percentage:.2f}%)"
        ]
        
        # Key Patterns
        if len(numeric_cols) > 0:
            correlations = self.cleaned_df[numeric_cols].corr()
            high_corr = [(col1, col2, corr) for col1 in correlations.columns 
                        for col2 in correlations.columns 
                        if col1 < col2 and abs(correlations.loc[col1, col2]) > 0.7]
            if high_corr:
                insights.append("\nStrong correlations found between:")
                for col1, col2, corr in high_corr[:3]:  # Show top 3 correlations
                    insights.append(f"- {col1} and {col2}: {corr:.2f}")
        
        # Data Distribution
        for col in numeric_cols[:2]:  # Analyze first 2 numeric columns
            mean_val = self.cleaned_df[col].mean()
            std_val = self.cleaned_df[col].std()
            insights.append(f"\n{col} distribution:")
            insights.append(f"- Mean: {mean_val:.2f}")
            insights.append(f"- Standard Deviation: {std_val:.2f}")
        
        return insights

    def generate_report(self):
        """Generate a PDF report with dataset analysis"""
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Dataset Analysis Report', ln=True, align='C')
        pdf.ln(10)
        
        # 1. Dataset Overview
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '1. Dataset Overview', ln=True)
        pdf.set_font('Arial', '', 12)
        
        # Original vs Cleaned Data
        pdf.cell(0, 10, f'Original Shape: {self.original_df.shape}', ln=True)
        pdf.cell(0, 10, f'Cleaned Shape: {self.cleaned_df.shape}', ln=True)
        pdf.ln(5)
        
        # Column Information
        pdf.cell(0, 10, 'Columns:', ln=True)
        for col in self.cleaned_df.columns:
            pdf.cell(0, 10, f'- {col}: {self.cleaned_df[col].dtype}', ln=True)
        pdf.ln(5)
        
        # 2. Data Quality Analysis
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '2. Data Quality Analysis', ln=True)
        pdf.set_font('Arial', '', 12)
        
        # Missing Values
        missing_original = self.original_df.isnull().sum().sum()
        missing_cleaned = self.cleaned_df.isnull().sum().sum()
        pdf.cell(0, 10, f'Missing Values (Original): {missing_original}', ln=True)
        pdf.cell(0, 10, f'Missing Values (Cleaned): {missing_cleaned}', ln=True)
        pdf.ln(5)
        
        # 3. Statistical Summary
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '3. Statistical Summary', ln=True)
        pdf.set_font('Arial', '', 12)
        
        # Get numeric columns
        numeric_cols = self.cleaned_df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            stats = self.cleaned_df[numeric_cols].describe()
            for col in numeric_cols:
                pdf.cell(0, 10, f'\nStatistics for {col}:', ln=True)
                for stat, value in stats[col].items():
                    pdf.cell(0, 10, f'- {stat}: {value:.2f}', ln=True)
                pdf.ln(5)
        
        # 4. Visualizations
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '4. Key Visualizations', ln=True)
        
        # Create and save visualizations
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Distribution plots for numeric columns
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                fig = px.histogram(self.cleaned_df, x=col, title=f'Distribution of {col}')
                temp_path = os.path.join(tmpdirname, f'{col}_dist.png')
                pio.write_image(fig, temp_path)
                pdf.image(temp_path, x=10, w=190)
                pdf.ln(5)
        
        # 5. Recommendations
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '5. Recommendations', ln=True)
        pdf.set_font('Arial', '', 12)
        
        recommendations = [
            "Consider feature engineering to create new meaningful variables",
            "Monitor data quality regularly",
            "Implement automated data validation checks",
            "Document any domain-specific data transformations",
            "Consider collecting additional relevant features"
        ]
        
        for rec in recommendations:
            pdf.cell(0, 10, f'- {rec}', ln=True)
        
        # 6. Executive Summary
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, '6. Executive Summary', ln=True)
        pdf.set_font('Arial', '', 12)
        
        summary_insights = self._generate_summary()
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Key Findings:', ln=True)
        pdf.set_font('Arial', '', 12)
        
        for insight in summary_insights:
            pdf.multi_cell(0, 10, insight)
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Future Improvements:', ln=True)
        pdf.set_font('Arial', '', 12)
        
        improvements = [
            "Implement automated data quality monitoring",
            "Develop feature engineering pipeline for key variables",
            "Create automated anomaly detection system",
            "Build interactive dashboards for real-time monitoring",
            "Establish regular data quality assessment procedures"
        ]
        
        for imp in improvements:
            pdf.cell(0, 10, f'- {imp}', ln=True)
        
        try:
            return pdf.output(dest='S').encode('latin1')
        except UnicodeEncodeError:
            # If encoding fails, try to remove problematic characters
            return pdf.output(dest='S').encode('latin1', errors='replace') 