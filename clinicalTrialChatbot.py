import streamlit as st
import pandas as pd
import os
import openai
import time
from transformers import pipeline

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
client = openai.OpenAI(api_key=api_key)

# Initialize Hugging Face pipeline for text classification
@st.cache_resource # prevents the model from being loaded multiple times.
def load_model():
    return pipeline("text-classification", model="bert-base-uncased")

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

# Function to extract information and filter data based on user query
def process_query(dataframe, user_question):
    """
    Analyzes the user query and filters the DataFrame based on extracted information.

    Args:
        dataframe (pd.DataFrame): The joined DataFrame.
        user_question (str): The user's question.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    try:
        # Use OpenAI to extract relevant information from the query
        prompt = f"""
        You are an expert in analyzing clinical trial data.
        The user will ask a question about the data.
        Your task is to extract the following information from the question:
        1.  The adverse event (AETERM). If no adverse event is mentioned, set to None.
        2.  The minimum age required (AGE) in integer format. If no age is mentioned, set to None.

        Here are some examples:
        User Question: How many patients are with adverse event APPLICATION SITE ERYTHEMA who are older than 75?
        Adverse Event: APPLICATION SITE ERYTHEMA
        Age: 75

        User Question: How many patients are older than 80?
        Adverse Event: None
        Age: 80

        User Question: How many patients are with adverse event Diarrhea?
        Adverse Event: Diarrhea
        Age: None

        User Question: How many patients are there?
        Adverse Event: None
        Age: None

        Now, extract the information from this question:
        {user_question}

        Adverse Event:
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at extracting information from clinical trial data questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        adverse_event = response.choices[0].message.content.strip()

        prompt = f"""
        You are an expert in analyzing clinical trial data.
        The user will ask a question about the data.
        Your task is to extract the following information from the question:
        1.  The adverse event (AETERM). If no adverse event is mentioned, set to None.
        2.  The minimum age required (AGE) in integer format. If no age is mentioned, set to None.

        Here are some examples:
        User Question: How many patients are with adverse event APPLICATION SITE ERYTHEMA who are older than 75?
        Adverse Event: APPLICATION SITE ERYTHEMA
        Age: 75

        User Question: How many patients are older than 80?
        Adverse Event: None
        Age: 80

        User Question: How many patients are with adverse event Diarrhea?
        Adverse Event: Diarrhea
        Age: None

        User Question: How many patients are there?
        Adverse Event: None
        Age: None

        Now, extract the information from this question:
        {user_question}

        Age:
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at extracting information from clinical trial data questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        age_str = response.choices[0].message.content.strip()

        age = None
        if age_str.isdigit():
            age = int(age_str)

        # Filter the DataFrame based on extracted information
        filtered_df = dataframe.copy() # Create a copy to avoid modifying the original DataFrame

        if adverse_event and adverse_event != "None":
            filtered_df = filtered_df[filtered_df['AETERM'] == adverse_event]
        if age is not None:
            filtered_df = filtered_df[filtered_df['AGE'] > age]

        return filtered_df

    except Exception as e:
        st.error(f"Error processing query: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Function to generate a response using OpenAI
def generate_response(filtered_data, adverse_event, age):
    """
    Generates a deterministic response to the user's input using OpenAI, based on filtered data.

    Args:
        filtered_data (pd.DataFrame): The filtered DataFrame.
        adverse_event (str): The adverse event extracted from the query.
        age (int): The age extracted from the query.

    Returns:
        str: The generated response.
    """
    count = len(filtered_data)
    if count > 0:
        response_str = f"There are {count} patients"
        if adverse_event and adverse_event != "None":
            response_str += f" with adverse event {adverse_event}"
        if age is not None:
            response_str += f" who are older than {age}"
        response_str += "."
        return response_str
    else:
        return "There are no patients matching the specified criteria."

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
                # Process the user query and filter the DataFrame
                filtered_data = process_query(joined_df, user_input)

                if not filtered_data.empty:
                    # Extract criteria from the processed query
                    adverse_event = extract_adverse_event(user_input)
                    age = extract_age(user_input)

                    count = len(filtered_data)
                    st.write(f"### Number of matching patients: {count}")
                    st.write("### Matching Patient Records:")
                    st.dataframe(filtered_data)

                    # Generate response using OpenAI for additional context or explanation
                    response = generate_response(filtered_data, adverse_event, age)
                    st.write("### Answer:")
                    st.write(response)
                else:
                    st.write("No matching records found for your query.")
                    
else:
    st.error("Failed to load data. Please check if 'ae - ae.csv' and 'dm - dm.csv' are present in the project folder.")

def extract_adverse_event(user_question):
    """
    Extracts the adverse event from the user's question using OpenAI.

    Args:
        user_question (str): The user's question.

    Returns:
        str: The adverse event extracted from the query.
    """
    try:
        prompt = f"""
        You are an expert in analyzing clinical trial data.
        The user will ask a question about the data.
        Your task is to extract the adverse event (AETERM) from the question.
        If no adverse event is mentioned, respond with "None".

        Here are some examples:
        User Question: How many patients are with adverse event APPLICATION SITE ERYTHEMA who are older than 75?
        Adverse Event: APPLICATION SITE ERYTHEMA

        User Question: How many patients are older than 80?
        Adverse Event: None

        Now, extract the adverse event from this question:
        {user_question}
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at extracting information from clinical trial data questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error extracting adverse event: {e}")
        return None

def extract_age(user_question):
    """
    Extracts the age from the user's question using OpenAI.

    Args:
        user_question (str): The user's question.

    Returns:
        int: The age extracted from the query.
    """
    try:
        prompt = f"""
        You are an expert in analyzing clinical trial data.
        The user will ask a question about the data.
        Your task is to extract the minimum age required (AGE) in integer format from the question.
        If no age is mentioned, respond with "None".

        Here are some examples:
        User Question: How many patients are with adverse event APPLICATION SITE ERYTHEMA who are older than 75?
        Age: 75

        User Question: How many patients are older than 80?
        Age: 80

        Now, extract the age from this question:
        {user_question}
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert at extracting information from clinical trial data questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        age_str = response.choices[0].message.content.strip()

        if age_str.isdigit():
            return int(age_str)
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting age: {e}")
        return None
