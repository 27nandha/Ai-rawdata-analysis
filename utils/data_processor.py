import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class DataProcessor:
    def __init__(self, df):
        self.df = df
        
    def get_basic_info(self):
        """Return basic information about the dataset"""
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        return info
    
    def create_visualization(self, plot_type, x_column, y_column=None, title=""):
        """Create different types of visualizations based on user selection"""
        if plot_type == "Histogram":
            fig = px.histogram(self.df, x=x_column, title=title)
            return fig
            
        elif plot_type == "Scatter Plot":
            fig = px.scatter(self.df, x=x_column, y=y_column, title=title)
            return fig
            
        elif plot_type == "Bar Plot":
            fig = px.bar(self.df, x=x_column, y=y_column, title=title)
            return fig
            
        elif plot_type == "Box Plot":
            fig = px.box(self.df, y=x_column, title=title)
            return fig
            
        elif plot_type == "Line Plot":
            fig = px.line(self.df, x=x_column, y=y_column, title=title)
            return fig
            
        elif plot_type == "Correlation Heatmap":
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = self.df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, title=title)
            return fig 