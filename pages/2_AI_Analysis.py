import streamlit as st
from utils.gemini_helper import GeminiAnalyzer
import asyncio

def main():
    st.title("🤖 AI Analysis")
    
    # Check if data is available
    if 'cleaned_data' not in st.session_state:
        st.error("Please upload and clean your data first in the Data Cleaning page!")
        return
    
    df = st.session_state['cleaned_data']
    
    # Create GeminiAnalyzer instance
    gemini_analyzer = GeminiAnalyzer(df)
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # AI Analysis Section
    st.subheader("Ask AI About Your Data")
    user_question = st.text_input("Ask a question about your data:")
    
    if user_question and st.button("Get AI Analysis"):
        with st.spinner("Analyzing..."):
            try:
                analysis = asyncio.run(gemini_analyzer.analyze_data(user_question))
                st.write(analysis)
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Please check your API key and try again.")

if __name__ == "__main__":
    main() 