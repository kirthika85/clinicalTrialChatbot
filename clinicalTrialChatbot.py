import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import time
import re

try:
    from transformers import pipeline
    import torch
except ImportError as e:
    st.error(f"Missing dependencies: {e}. Please install the required libraries with 'pip install transformers torch'")
    st.stop()

# Set up OpenAI API using environment variable
with st.spinner("🔄 Payer agent Authentication In progress..."):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ API_KEY not found in environment variables.")
        st.stop()
    time.sleep(5)
st.success("✅ Payer agent Authentication Successful")

if api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize Hugging Face pipeline for text classification
@st.cache_resource # prevents the model from being loaded multiple times.
def load_model():
    try:
        return pipeline("text-classification", model="bert-base-uncased", device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        st.error(f"Error loading model: {e}. Please make sure you have the correct dependencies installed (TensorFlow or PyTorch).")
        st.stop()

classifier = load_model()

# Function to read the CSV files and create the DataFrames
def create_dataframes():
    """
    Reads the AE and DM CSV files from the project folder, creates two DataFrames, and joins them.

    Returns:
        pandas.DataFrame: The joined DataFrame.
    """
    try:
        ae_df = pd.read_csv('ae - ae.csv')
        dm_df = pd.read_csv('dm - dm.csv')

        # Join the DataFrames using USUBJID
        joined_df = pd.merge(ae_df, dm_df, on='USUBJID', how='inner')  # Inner join to match patients in both datasets
        return joined_df

    except FileNotFoundError as e:
        st.error(f"Error: One or more files were not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading or processing the CSV files: {e}")
        return None

# Function to convert natural language to pandas query string
def generate_pandas_query(user_question):
    """
    Converts natural language into a pandas query string.
    Args:
        user_question (str): The user's question in natural language.
    Returns:
        str: A pandas query string.
    """
    try:
        # Use OpenAI to convert natural language into a pandas query string
        prompt = f"""
        You are an expert in converting natural language questions into pandas query strings.
        Your task is to convert the user's question into a pandas query string that can be used to filter a DataFrame.
        Here are some examples:

        User Question: How many patients are with adverse events DIARRHOEA and age above 75?
        Pandas Query: AETERM == 'DIARRHOEA' and AGE > 75

        User Question: How many patients are with adverse events APPLICATION SITE ERYTHEMA and WHITE race?
        Pandas Query: AETERM == 'APPLICATION SITE ERYTHEMA' and RACE == 'WHITE'

        User Question: How many male patients are older than 80 with fatigue?
        Pandas Query: SEX == 'M' and AGE > 80 and AETERM == 'FATIGUE'

        User Question: How many patients are older than 60 who are BLACK OR AFRICAN AMERICAN?
        Pandas Query: AGE > 60 and RACE == 'BLACK OR AFRICAN AMERICAN'

        User Question: How many patients are there?
        Pandas Query: None

        Now, convert this question:
        {user_question}
        Pandas Query:
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at converting natural language questions into pandas query strings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        query = response.choices[0].message.content.strip()

        return query

    except Exception as e:
        st.error(f"Error generating pandas query: {e}")
        return None

# Function to filter the DataFrame based on the generated query
def filter_data(dataframe, query):
    """
    Filters the DataFrame based on the generated query.
    Args:
        dataframe (pd.DataFrame): The joined DataFrame.
        query (str): The pandas query string.
    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    try:
        if query and query.lower() != 'none':
            filtered_df = dataframe.query(query)
            return filtered_df
        else:
            return dataframe  # If query is None, return the original DataFrame
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return pd.DataFrame()

# Function to generate a response using OpenAI
def generate_response(filtered_data):
    """
    Generates a deterministic response to the user's input using OpenAI, based on filtered data.
    Args:
        filtered_data (pd.DataFrame): The filtered DataFrame.
    Returns:
        str: The generated response.
    """
    count = len(filtered_data)
    return f"There are {count} patients matching the specified criteria." if count > 0 else "There are no patients matching the specified criteria."

# Streamlit App Layout
st.title("Clinical Trial Data Analyzer")

# Create the DataFrames
joined_df = create_dataframes()

if joined_df is not None:
    st.write("### Joined DataFrame (Sample):")
    st.dataframe(joined_df.head(10))

    # User Input
    user_input = st.text_input("Ask a question about the data:")
    
    if st.button("Get Answer"):
        if user_input:
            with st.spinner("Processing your query..."):
                # Generate pandas query
                pandas_query = generate_pandas_query(user_input)

                if pandas_query is not None:
                    st.write(f"Pandas Query: {pandas_query}")

                    # Filter the DataFrame
                    filtered_data = filter_data(joined_df, pandas_query)

                    if not filtered_data.empty:
                        count = len(filtered_data)
                        st.write(f"### Number of matching patients: {count}")
                        st.write("### Matching Patient Records:")
                        st.dataframe(filtered_data)

                        # Generate response using OpenAI for additional context or explanation
                        response = generate_response(filtered_data)
                        st.write("### Answer:")
                        st.write(response)
                    else:
                        st.write("No matching records found for your query.")
                else:
                    st.error("Failed to generate a valid query.")
                    
else:
    st.error("Failed to load data. Please check if 'ae - ae.csv' and 'dm - dm.csv' are present in the project folder.")
