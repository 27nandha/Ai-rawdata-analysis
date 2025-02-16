import streamlit as st
from utils.data_processor import DataProcessor
from utils.gemini_helper import GeminiAnalyzer

def main():
    st.title("📊 Data Visualization")
    
    # Check if data is available
    if 'cleaned_data' not in st.session_state:
        st.error("Please upload and clean your data first in the Data Cleaning page!")
        return
    
    df = st.session_state['cleaned_data']
    
    # Create processor instances
    processor = DataProcessor(df)
    gemini_analyzer = GeminiAnalyzer(df)
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Column selection for visualization suggestion
    st.subheader("Get Visualization Suggestions")
    selected_columns = st.multiselect(
        "Select columns for visualization suggestion",
        df.columns
    )
    
    if selected_columns and st.button("Get Suggestions"):
        with st.spinner("Generating suggestions..."):
            suggestions = gemini_analyzer.suggest_visualization(selected_columns)
            st.write(suggestions)
    
    # Manual visualization creation
    st.subheader("Create Custom Visualization")
    plot_types = ["Histogram", "Scatter Plot", "Bar Plot", "Box Plot", 
                 "Line Plot", "Correlation Heatmap"]
    plot_type = st.selectbox("Select the type of plot", plot_types)
    
    # Select columns for plotting
    if plot_type in ["Scatter Plot", "Bar Plot", "Line Plot"]:
        x_column = st.selectbox("Select X-axis column", df.columns)
        y_column = st.selectbox("Select Y-axis column", df.columns)
    else:
        x_column = st.selectbox("Select column", df.columns)
        y_column = None
    
    # Plot title
    title = st.text_input("Enter plot title", "")
    
    # Create visualization button
    if st.button("Generate Plot"):
        st.subheader("Visualization")
        fig = processor.create_visualization(plot_type, x_column, y_column, title)
        st.plotly_chart(fig)

if __name__ == "__main__":
    main() 