import streamlit as st
import pandas as pd
import os
import openai
import time

# Set up OpenAI API using environment variable
with st.spinner("🔄 Payer agent Authentication In progress..."):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("❌ API_KEY not found in environment variables.")
        st.stop()
    time.sleep(5)
st.success("✅ Payer agent Authentication Successful")

if openai.api_key is None:
    st.error("OPENAI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

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

# Function to generate a response using OpenAI
def generate_response(user_input, dataframe):
    """
    Generates a deterministic response to the user's input using OpenAI, based on the DataFrame.

    Args:
        user_input (str): The user's question.
        dataframe (pandas.DataFrame): The joined DataFrame.

    Returns:
        str: The generated response.
    """
    try:
        # Construct the prompt for OpenAI
        prompt = f"""
        You are an expert at analyzing clinical trial data and providing precise, deterministic answers.
        Here is the data you can analyze:
        ```
        {dataframe.to_string(index=False, max_rows=20)}
        ```
        Follow these rules:
        1.  Answer the user's question using ONLY the data provided.  Do NOT make assumptions or use external knowledge.
        2.  If the data is sufficient to answer, provide the answer and ALWAYS include the count of the patients related to the answer.
        3.  If the data is NOT sufficient to answer the question with certainty, respond with "I cannot answer this question with the available data." and do not include any patient counts.

        User question: {user_input}
        """

        # Call OpenAI's ChatCompletion API
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a clinical trial data expert focused on deterministic answers."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.0,  # Make results deterministic!
            seed = 42, # Added seed for reproducibility
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit App Layout
st.title("Clinical Trial Data Analyzer")

# Create the DataFrames
joined_df = create_dataframes()

if joined_df is not None:
    # Display the DataFrame (optional - for debugging)
    st.write("### Joined DataFrame (Sample):")
    st.dataframe(joined_df.head(10))

    # User Input
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
    st.error("Failed to load data. Please check if 'ae.csv' and 'dm.csv' are present in the project folder.")
