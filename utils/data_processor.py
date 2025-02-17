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
            
        elif plot_type == "Scatter Plot":
            fig = px.scatter(self.df, x=x_column, y=y_column, title=title)
            
        elif plot_type == "Bar Chart":
            if y_column:
                # If y_column provided, show average of y grouped by x
                data = self.df.groupby(x_column)[y_column].mean().reset_index()
                fig = px.bar(data, x=x_column, y=y_column, title=title)
            else:
                # If no y_column, show counts of x categories
                value_counts = self.df[x_column].value_counts()
                data = pd.DataFrame({
                    'category': value_counts.index,
                    'count': value_counts.values
                })
                fig = px.bar(data, x='category', y='count', title=title)
            
        elif plot_type == "Line Chart":
            # Sort by x_column if it's time-based
            temp_df = self.df.sort_values(x_column)
            fig = px.line(temp_df, x=x_column, y=y_column, title=title)
            
        elif plot_type == "Pie Chart":
            # Create value counts and prepare data for pie chart
            value_counts = self.df[x_column].value_counts()
            data = pd.DataFrame({
                'category': value_counts.index,
                'count': value_counts.values
            })
            fig = px.pie(data, values='count', names='category', title=title)
        
        # Apply consistent template
        fig.update_layout(template='plotly_white')
        return fig 