import streamlit as st
import os

def main():
    st.set_page_config(
        page_title="Data Analysis & Visualization App",
        page_icon="📊",
        layout="wide"
    )
    
    # Home page content
    st.title("🎯 Welcome to Data Analysis & Visualization App")
    
    st.markdown("""
    ### 🚀 About This App
    This application helps you analyze and visualize your data using AI-powered insights. 
    Upload your CSV file and explore various features across different pages.
    
    ### 📌 Main Features
    
    1. **Data Cleaning** 🧹
        * Upload and preview your dataset
        * Handle missing values
        * Remove duplicates
        * Fix data types
        * Download cleaned data
    
    2. **AI Analysis** 🤖
        * Ask questions about your data
        * Get AI-powered insights
        * Search through your dataset
        * Analyze trends and patterns
    
    3. **Data Visualization** 📊
        * Create various types of plots
        * Get AI suggestions for visualizations
        * Customize your visualizations
        * Export plots
    
    ### 🎯 How to Use
    1. Start by uploading your CSV file in the Data Cleaning page
    2. Clean and prepare your data
    3. Use the AI Analysis page to explore your data
    4. Create visualizations to better understand your data
    
    ### 💡 Tips
    * Make sure your CSV file is properly formatted
    * Clean your data before analysis
    * Be specific with your questions to the AI
    * Try different visualization types for better insights
    """)
    
    # Navigation buttons
    st.markdown("### 🚀 Quick Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.button("🧹 Data Cleaning", 
                 on_click=lambda: st.switch_page("pages/1_Data_Cleaning.py"))
    with col2:
        st.button("🤖 AI Analysis", 
                 on_click=lambda: st.switch_page("pages/2_AI_Analysis.py"))
    with col3:
        st.button("📊 Visualization", 
                 on_click=lambda: st.switch_page("pages/3_Visualization.py"))

if __name__ == "__main__":
    main() 