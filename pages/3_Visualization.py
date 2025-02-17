import streamlit as st
from utils.data_processor import DataProcessor
from utils.gemini_helper import GeminiAnalyzer
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def get_ai_visualization_suggestions(df, analyzer):
    """Get AI suggestions for visualizations based on data analysis"""
    prompt = f"""
    Analyze this dataset carefully and suggest 3-4 most insightful visualizations using ONLY these chart types:
    - Line Chart (for trends over time)
    - Bar Chart (for comparisons)
    - Pie Chart (for composition)
    - Histogram (for distributions)
    - Scatter Plot (for relationships)

    Dataset Information:
    - Shape: {df.shape}
    - Columns: {list(df.columns)}
    - Numeric Columns: {list(df.select_dtypes(include=['float64', 'int64']).columns)}
    - Categorical Columns: {list(df.select_dtypes(include=['object', 'category']).columns)}
    - Sample Data:
    {df.head().to_string()}

    For each visualization, provide:
    1. Chart Type (from the list above)
    2. Specific columns to use
    3. What insights this visualization will reveal
    
    YOU MUST SUGGEST EXACTLY 4 VISUALIZATIONS.
    """
    
    try:
        suggestions = analyzer.suggest_visualization(df.columns, custom_prompt=prompt)
        return suggestions
    except Exception as e:
        return f"Error getting visualization suggestions: {str(e)}"

def create_visualizations(df, processor):
    """Create a set of meaningful visualizations"""
    plots = []
    
    # Get column types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # 1. Histogram for the most important numeric column
    if len(numeric_cols) > 0:
        main_numeric = numeric_cols[0]
        fig = px.histogram(df, x=main_numeric, 
                          title=f'Distribution of {main_numeric}',
                          template='plotly_white')
        plots.append(("Distribution Analysis", fig, 
                     f"Shows the frequency distribution of {main_numeric}"))
    
    # 2. Bar Chart for categorical data
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        fig = px.bar(df.groupby(cat_col)[num_col].mean().reset_index(), 
                    x=cat_col, y=num_col,
                    title=f'Average {num_col} by {cat_col}',
                    template='plotly_white')
        plots.append(("Categorical Comparison", fig,
                     f"Compares average {num_col} across different {cat_col} categories"))
    
    # 3. Line Chart for time-based data or sequence
    time_cols = [col for col in df.columns if any(term in col.lower() 
                for term in ['date', 'year', 'month', 'time'])]
    if time_cols and len(numeric_cols) > 0:
        time_col = time_cols[0]
        num_col = numeric_cols[0]
        # Sort by time column
        temp_df = df.sort_values(time_col)
        fig = px.line(temp_df, x=time_col, y=num_col,
                     title=f'Trend of {num_col} over {time_col}',
                     template='plotly_white')
        plots.append(("Trend Analysis", fig,
                     f"Shows how {num_col} changes over {time_col}"))
    
    # 4. Scatter Plot for relationship between numeric columns
    if len(numeric_cols) >= 2:
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                        title=f'Relationship: {numeric_cols[0]} vs {numeric_cols[1]}',
                        template='plotly_white')
        plots.append(("Correlation Analysis", fig,
                     f"Shows the relationship between {numeric_cols[0]} and {numeric_cols[1]}"))
    
    # Alternative: Pie Chart if we don't have enough numeric columns
    elif len(categorical_cols) > 0:
        cat_col = categorical_cols[0]
        # Create value counts and prepare data for pie chart
        value_counts = df[cat_col].value_counts()
        pie_data = pd.DataFrame({
            'category': value_counts.index,
            'count': value_counts.values
        })
        fig = px.pie(pie_data, 
                    values='count',
                    names='category',
                    title=f'Distribution of {cat_col}',
                    template='plotly_white')
        plots.append(("Composition Analysis", fig,
                     f"Shows the distribution of {cat_col} categories"))
    
    return plots

def main():
    st.title("📊 Data Visualization")
    
    if 'cleaned_data' not in st.session_state:
        st.error("Please upload and clean your data first in the Data Cleaning page!")
        return
    
    df = st.session_state['cleaned_data']
    processor = DataProcessor(df)
    gemini_analyzer = GeminiAnalyzer(df)
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Generate Visualizations
    st.subheader("🎯 Key Data Insights")
    
    with st.spinner("Generating visualizations..."):
        try:
            # Create visualizations first
            plots = create_visualizations(df, processor)
            
            # Get AI suggestions
            suggestions = get_ai_visualization_suggestions(df, gemini_analyzer)
            
            # Show AI insights
            st.write("### 💡 AI Analysis")
            st.write(suggestions)
            
            # Display visualizations
            st.write("### 📊 Visual Insights")
            for title, fig, explanation in plots:
                st.write(f"**{title}**")
                st.write(f"*{explanation}*")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
        except Exception as e:
            st.error(f"Error generating visualizations: {str(e)}")
    
    # Manual visualization creation
    st.subheader("🎨 Create Custom Visualization")
    plot_types = ["Line Chart", "Bar Chart", "Pie Chart", "Histogram", "Scatter Plot"]
    plot_type = st.selectbox("Select the type of plot", plot_types)
    
    try:
        # Select columns based on plot type
        if plot_type in ["Line Chart", "Bar Chart", "Scatter Plot"]:
            x_column = st.selectbox("Select X-axis column", df.columns)
            y_column = st.selectbox("Select Y-axis column", df.columns)
        elif plot_type == "Pie Chart":
            x_column = st.selectbox("Select category column", 
                                  df.select_dtypes(include=['object', 'category']).columns)
            y_column = None
        else:  # Histogram
            x_column = st.selectbox("Select column", 
                                  df.select_dtypes(include=['float64', 'int64']).columns)
            y_column = None
        
        title = st.text_input("Enter plot title", "")
        
        if st.button("Generate Plot"):
            st.subheader("Visualization")
            fig = processor.create_visualization(plot_type, x_column, y_column, title)
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

if __name__ == "__main__":
    main() 