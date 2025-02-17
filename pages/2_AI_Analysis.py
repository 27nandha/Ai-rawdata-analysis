import streamlit as st
from utils.gemini_helper import GeminiAnalyzer
import asyncio

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

async def get_ai_response(analyzer, question):
    try:
        response = await analyzer.analyze_data(question)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    st.title("🤖 AI Data Analysis Chat")
    
    # Check if data is available
    if 'cleaned_data' not in st.session_state:
        st.error("Please upload and clean your data first in the Data Cleaning page!")
        return
    
    df = st.session_state['cleaned_data']
    
    # Initialize chat history
    initialize_chat_history()
    
    # Create GeminiAnalyzer instance
    gemini_analyzer = GeminiAnalyzer(df)
    
    # Show data preview in an expander
    with st.expander("📊 View Data Preview", expanded=False):
        st.dataframe(df.head())
        st.write(f"Total Records: {len(df)}")
        st.write(f"Columns: {', '.join(df.columns)}")
    
    # Chat interface
    st.markdown("### 💬 Chat with Your Data")
    
    # Example questions
    with st.expander("❓ Example Questions", expanded=False):
        st.markdown("""
        Try asking questions like:
        - What is the total number of records?
        - Show me the distribution of [column_name]
        - What are the trends in the data?
        - Find the highest and lowest values in [column_name]
        - Compare [column1] with [column2]
        - Search for records where [condition]
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.write(content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        add_message("user", prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = asyncio.run(get_ai_response(gemini_analyzer, prompt))
                st.write(response)
        add_message("assistant", response)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Sidebar with additional information
    st.sidebar.markdown("""
    ### 📝 Tips for Better Results
    1. Be specific in your questions
    2. Mention column names correctly
    3. Ask one question at a time
    4. Use clear comparisons
    5. Specify time periods if relevant
    
    ### 🎯 Available Analysis Types
    - Statistical Analysis
    - Trend Detection
    - Pattern Recognition
    - Data Comparison
    - Search & Filter
    - Distribution Analysis
    """)
    
    # Download chat history
    if st.sidebar.button("📥 Download Chat History"):
        chat_text = "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in st.session_state.messages
        ])
        st.sidebar.download_button(
            label="Save Chat",
            data=chat_text,
            file_name="data_analysis_chat.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main() 