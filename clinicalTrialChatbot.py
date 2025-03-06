import streamlit as st
import pandas as pd
import os
from openai import OpenAI
import time

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

# Function to read the CSV files and create the DataFrames
def create_dataframes():
    try:
        ae_df = pd.read_csv('ae - ae.csv')
        dm_df = pd.read_csv('dm - dm.csv')

        st.write(f"AE DataFrame shape: {ae_df.shape}")
        st.write(f"DM DataFrame shape: {dm_df.shape}")

        st.write("Sample of AE DataFrame:")
        st.dataframe(ae_df.head())

        st.write("Sample of DM DataFrame:")
        st.dataframe(dm_df.head())

        ae_subjects = set(ae_df['USUBJID'])
        dm_subjects = set(dm_df['USUBJID'])
        common_subjects = ae_subjects.intersection(dm_subjects)

        st.write(f"Unique subjects in AE: {len(ae_subjects)}")
        st.write(f"Unique subjects in DM: {len(dm_subjects)}")
        st.write(f"Common subjects: {len(common_subjects)}")

        joined_df = pd.merge(ae_df, dm_df, on='USUBJID', how='outer')
        
        st.write(f"Joined DataFrame shape: {joined_df.shape}")
        st.write("Sample of Joined DataFrame:")
        st.dataframe(joined_df.head(20))

        return joined_df

    except FileNotFoundError as e:
        st.error(f"Error: One or more files were not found: {e}")
        return None
    except Exception as e:
        st.error(f"Error reading or processing the CSV files: {e}")
        return None

# Function to generate a response using OpenAI
def generate_response(user_input, dataframe):
    try:
        prompt = f"""
        You are an expert at analyzing clinical trial data and providing precise, deterministic answers.
        Here is the data you can analyze:
        ```
        {dataframe.to_string(index=False, max_rows=20)}
        ```
        Follow these rules:
        1. Answer the user's question using ONLY the data provided. Do NOT make assumptions or use external knowledge.
        2. If the data is sufficient to answer, provide the answer and ALWAYS include the count of the patients related to the answer.
        3. If the data is NOT sufficient to answer the question with certainty, respond with "I cannot answer this question with the available data." and do not include any patient counts.

        User question: {user_input}
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a clinical trial data expert focused on deterministic answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            seed=42
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App Layout
st.title("Clinical Trial Data Analyzer")

# Create the DataFrames
joined_df = create_dataframes()

if joined_df is not None:
    st.write("### Joined DataFrame (Sample):")
    st.dataframe(joined_df.head(10))

    user_input = st.text_input("Ask a question about the data:")

    if st.button("Get Answer"):
        if user_input:
            with st.spinner("Generating response..."):
                response = generate_response(user_input, joined_df)
                st.write("### Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")
else:
    st.error("Failed to load data. Please check if 'ae - ae.csv' and 'dm - dm.csv' are present in the project folder.")
