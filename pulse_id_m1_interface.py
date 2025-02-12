__import__('pysqlite3')
import sys
import os
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from groq import Groq
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import re
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process, LLM

# Set up Groq API
GROQ_API_KEY = "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"


llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

# Streamlit app
st.set_page_config(
    page_title="📊 CSV Data Analysis with CrewAI and Groq",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
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
                Provider = 'Groq',
                llm = llm
            )

            data_analyst_agent = Agent(
                role="Data Analyst",
                goal="Analyze the extracted data and provide meaningful insights.",
                backstory="You are a data analyst with expertise in deriving insights from structured data.",
                Provider = 'Groq',
                llm = llm
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
                process=Process.sequential
            )

            # Execute the crew
            result = crew.kickoff()

            # Display the results
            st.subheader("Extracted Data:")
            st.write(result)  # Output from the Data Extractor Agent

            st.subheader("Analysis and Insights:")
            st.write(result["analyze_task_output"])  # Output from the Data Analyst Agent
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")
