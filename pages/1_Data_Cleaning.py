import streamlit as st
import pandas as pd
import io

def clean_data(df):
    """Basic data cleaning operations"""
    # Create a copy of the dataframe
    cleaned_df = df.copy()
    
    # Get user input for cleaning operations
    st.subheader("Cleaning Operations")
    
    # Handle missing values
    if st.checkbox("Handle Missing Values"):
        missing_strategy = st.selectbox(
            "Choose strategy for missing values:",
            ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with custom value"]
        )
        
        if missing_strategy == "Drop rows":
            cleaned_df = cleaned_df.dropna()
        elif missing_strategy == "Fill with mean":
            cleaned_df = cleaned_df.fillna(cleaned_df.mean(numeric_only=True))
        elif missing_strategy == "Fill with median":
            cleaned_df = cleaned_df.fillna(cleaned_df.median(numeric_only=True))
        elif missing_strategy == "Fill with mode":
            cleaned_df = cleaned_df.fillna(cleaned_df.mode().iloc[0])
        elif missing_strategy == "Fill with custom value":
            custom_value = st.text_input("Enter custom value:")
            if custom_value:
                cleaned_df = cleaned_df.fillna(custom_value)
    
    # Remove duplicates
    if st.checkbox("Remove Duplicate Rows"):
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Fix data types
    if st.checkbox("Fix Data Types"):
        for column in cleaned_df.columns:
            current_type = cleaned_df[column].dtype
            new_type = st.selectbox(
                f"Select type for {column} (current: {current_type})",
                ["int64", "float64", "string", "datetime64[ns]", "category", "Keep current"],
                index=5
            )
            if new_type != "Keep current":
                try:
                    if new_type == "datetime64[ns]":
                        cleaned_df[column] = pd.to_datetime(cleaned_df[column])
                    else:
                        cleaned_df[column] = cleaned_df[column].astype(new_type)
                except Exception as e:
                    st.error(f"Error converting {column}: {str(e)}")
    
    return cleaned_df

def main():
    st.title("🧹 Data Cleaning")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Show original data
        st.subheader("Original Data Preview")
        st.dataframe(df.head())
        
        # Show data info
        st.subheader("Data Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Clean data
        cleaned_df = clean_data(df)
        
        # Show cleaned data
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head())
        
        # Download cleaned data
        if st.button("Download Cleaned Data"):
            csv = cleaned_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )
        
        # Store cleaned data in session state for other pages
        st.session_state['cleaned_data'] = cleaned_df

if __name__ == "__main__":
    main() 