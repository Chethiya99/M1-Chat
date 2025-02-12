import streamlit as st
import pandas as pd
#from crewai import Agent, Task, Crew
from crewai import Agent, Task, Crew, Process, LLM
from groq import Groq

# Set up Groq API
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

# Streamlit app
st.title("📊 CSV Data Analysis with CrewAI and Groq")
st.markdown("Upload a CSV file, enter your query, and get insights!")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of the uploaded CSV file:")
    st.write(df.head())

    # User query input
    user_query = st.text_input("Enter your query (e.g., 'Show sales data for 2023'):")
    if st.button("Run Query"):
        if user_query:
            # Define the CrewAI agents
            data_extractor_agent = Agent(
                role="Data Extractor",
                goal=f"Extract relevant data from the CSV based on the user query: {user_query}",
                backstory="You are an expert at extracting specific data from datasets based on user queries.",
                tools=[],  # No tools needed for this example
                verbose=True
            )

            data_analyst_agent = Agent(
                role="Data Analyst",
                goal="Analyze the extracted data and provide meaningful insights.",
                backstory="You are a data analyst with expertise in deriving insights from structured data.",
                tools=[],  # No tools needed for this example
                verbose=True
            )

            # Define tasks for the agents
            extract_task = Task(
                description=f"Extract data from the CSV based on the query: {user_query}",
                agent=data_extractor_agent,
                expected_output="A subset of the CSV data relevant to the user's query."
            )

            analyze_task = Task(
                description="Analyze the extracted data and provide insights.",
                agent=data_analyst_agent,
                expected_output="A detailed analysis of the extracted data with actionable insights."
            )

            # Create the crew and execute tasks
            crew = Crew(
                agents=[data_extractor_agent, data_analyst_agent],
                tasks=[extract_task, analyze_task],
                verbose=2
            )

            # Execute the crew
            result = crew.kickoff()

            # Display the results
            st.subheader("Extracted Data:")
            st.write(result["extract_task_output"])  # Output from the Data Extractor Agent

            st.subheader("Analysis and Insights:")
            st.write(result["analyze_task_output"])  # Output from the Data Analyst Agent
