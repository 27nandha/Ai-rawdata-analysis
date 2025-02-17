from google import genai
import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

class GeminiAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df_info = self._get_df_info()
        
        # Try to get API key from Streamlit secrets first, then from environment
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except:
            load_dotenv()
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables or Streamlit secrets")
        
        self.client = genai.Client(api_key=api_key)

    def _get_df_info(self):
        """Get basic dataset information for context"""
        info = f"""
        Dataset Information:
        - Shape: {self.df.shape}
        - Columns: {list(self.df.columns)}
        - Data Types: {self.df.dtypes.to_dict()}
        - Sample Data: 
{self.df.head(5).to_string()}
        
        Summary Statistics:
{self.df.describe().to_string()}
        """
        return info

    def _process_data_for_question(self, question):
        """Process data based on the question"""
        question_lower = question.lower()
        result = ""

        try:
            # Search functionality
            if "search" in question_lower or "find" in question_lower:
                search_terms = []
                # Extract potential search terms from the question
                words = question_lower.split()
                for i, word in enumerate(words):
                    if word in ["search", "find", "look", "show"] and i + 1 < len(words):
                        search_terms.extend(words[i+1:])
                
                if search_terms:
                    # Search across all columns
                    matches = pd.DataFrame()
                    for column in self.df.columns:
                        # Convert column to string for searching
                        column_matches = self.df[self.df[column].astype(str).str.contains('|'.join(search_terms), case=False, na=False)]
                        matches = pd.concat([matches, column_matches]).drop_duplicates()
                    
                    if not matches.empty:
                        result += f"\nSearch Results:\n{matches.to_string()}"
                        result += f"\nTotal matches found: {len(matches)}"
                    else:
                        result += "\nNo matches found."

            # Year and customer analysis
            elif "year" in question_lower and "customer" in question_lower:
                if 'year' in self.df.columns:
                    yearly_counts = self.df['year'].value_counts().sort_index()
                    result += f"\nYearly Customer Distribution:\n{yearly_counts.to_string()}"
                    result += f"\nTotal Customers: {len(self.df)}"
                    
                    # Add year-over-year growth
                    if len(yearly_counts) > 1:
                        yoy_growth = yearly_counts.pct_change() * 100
                        result += f"\n\nYear-over-Year Growth (%):\n{yoy_growth.to_string()}"

            # Customer analysis
            elif "customer" in question_lower:
                result += f"\nTotal number of customers: {len(self.df)}"
                
                # Add customer demographics if available
                demographic_cols = [col for col in self.df.columns if any(term in col.lower() 
                                 for term in ['age', 'gender', 'location', 'country', 'city'])]
                
                for col in demographic_cols:
                    value_counts = self.df[col].value_counts()
                    result += f"\n\n{col} Distribution:\n{value_counts.to_string()}"

            # Statistical analysis
            elif any(term in question_lower for term in ['average', 'mean', 'median', 'max', 'min', 'sum']):
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                
                if numeric_cols.empty:
                    result += "\nNo numeric columns found for statistical analysis."
                else:
                    stats = self.df[numeric_cols].describe()
                    result += f"\nStatistical Summary:\n{stats.to_string()}"

            # Trend analysis
            elif "trend" in question_lower or "pattern" in question_lower:
                numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
                if 'date' in self.df.columns or 'year' in self.df.columns:
                    time_col = 'date' if 'date' in self.df.columns else 'year'
                    for col in numeric_cols:
                        trend = self.df.groupby(time_col)[col].mean()
                        result += f"\n\nTrend for {col}:\n{trend.to_string()}"

            # If no specific analysis is triggered, provide general dataset info
            if not result:
                result += f"\nDataset Overview:"
                result += f"\n- Total Records: {len(self.df)}"
                result += f"\n- Columns: {', '.join(self.df.columns)}"
                result += f"\n- Missing Values: {self.df.isnull().sum().to_dict()}"
                
                # Sample of unique values for each column
                for col in self.df.columns:
                    unique_vals = self.df[col].nunique()
                    result += f"\n\n{col} - Unique Values: {unique_vals}"
                    if unique_vals < 10:  # Only show distribution for categorical-like columns
                        result += f"\nDistribution:\n{self.df[col].value_counts().head().to_string()}"

            return result
        except Exception as e:
            return f"Error processing data: {str(e)}"

    async def analyze_data(self, question):
        """Analyze the data based on user question"""
        try:
            # Process data specifically for the question
            data_analysis = self._process_data_for_question(question)
            
            prompt = f"""
            Given the following dataset information:
            {self.df_info}
            
            Question: {question}
            
            Additional Data Analysis:
            {data_analysis}
            
            Please provide a detailed and accurate analysis based on the actual data shown above.
            Make sure to:
            1. Use the exact numbers from the data
            2. Include all relevant years/categories
            3. Double-check totals and calculations
            4. Mention any important patterns or trends
            5. Note if any data is missing or incomplete
            """
            
            response = self.client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            
            # Return only the AI response without the data summary
            return response.text
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    def suggest_visualization(self, columns):
        """Suggest appropriate visualization types for given columns"""
        try:
            # Get data types and sample statistics for selected columns
            column_stats = {
                col: {
                    'dtype': str(self.df[col].dtype),
                    'unique_values': len(self.df[col].unique()),
                    'has_nulls': self.df[col].isnull().any(),
                    'sample_values': self.df[col].head(3).tolist()
                } for col in columns
            }
            
            prompt = f"""
            Given these columns from the dataset: {columns}
            Column Details: {column_stats}
            
            Suggest the most appropriate type of visualization(s) from these options:
            - Histogram
            - Scatter Plot
            - Bar Plot
            - Box Plot
            - Line Plot
            - Correlation Heatmap
            
            For each suggested visualization:
            1. Explain why it's appropriate for this data
            2. What insights it could reveal
            3. Any potential limitations or considerations
            """
            
            response = self.client.models.generate_content(
                model="gemini-pro",
                contents=prompt
            )
            return response.text
            
        except Exception as e:
            return f"Error suggesting visualization: {str(e)}" 