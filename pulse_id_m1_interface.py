__import__('pysqlite3')
import sys
import os
import sqlite3
import re
import pandas as pd
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents import AgentType
from crewai import Agent, Task, Crew, Process, LLM

# Page Configuration
st.set_page_config(
    page_title="Pulse iD - Database Query & Insights Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'merchant_data' not in st.session_state:
    st.session_state.merchant_data = None
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = ""
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'interaction_history' not in st.session_state:
    st.session_state.interaction_history = []  # Store all interactions (queries, results, insights)
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = "merchant_data_singapore.db"  # Default database
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False  # Track if the database is initialized
if 'trigger_rerun' not in st.session_state:
    st.session_state.trigger_rerun = False  # Track if a re-run is needed

# Header Section with Title and Logo
st.image("logo.png", width=150)  # Ensure you have your logo in the working directory
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>📊 PulseID Merchant Scout Agent</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center; color: #555;'>Interact with your merchant database and gain insights with ease!</h4>",
    unsafe_allow_html=True
)

# Sidebar Configuration
st.sidebar.header("Settings")

def get_api_key():
    """Function to get API Key from user input"""
    return st.sidebar.text_input("Enter Your API Key:", type="password")

# Get API Key
api_key = get_api_key()
if api_key:
    st.session_state.api_key = api_key

# Database Selection
db_options = ["merchant_data_dubai.db", "merchant_data_singapore.db"]
new_selected_db = st.sidebar.selectbox("Select Database:", db_options, index=db_options.index(st.session_state.selected_db))

# Check if the database selection has changed
if new_selected_db != st.session_state.selected_db:
    st.session_state.selected_db = new_selected_db
    st.session_state.db_initialized = False  # Reset database initialization
    st.sidebar.success(f"✅ Switched to database: {st.session_state.selected_db}")

# Model Selection
model_name = st.sidebar.selectbox("Select Model:", ["llama3-70b-8192"])  # "llama3-70b-8192"

# Initialize SQL Database and Agent
if st.session_state.selected_db and api_key and not st.session_state.db_initialized:
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            temperature=0,
            model_name=model_name,
            api_key=st.session_state.api_key
        )

        # Initialize SQLDatabase
        st.session_state.db = SQLDatabase.from_uri(f"sqlite:///{st.session_state.selected_db}", sample_rows_in_table_info=3)

        # Create SQL Agent
        st.session_state.agent_executor = create_sql_agent(
            llm=llm,
            db=st.session_state.db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        st.session_state.db_initialized = True  # Mark database as initialized
        st.sidebar.success("✅ Database and LLM Connected Successfully!")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

# Function to render the "Enter Query" section
def render_query_section():
    st.markdown("#### Get to know the Merchant Target List:", unsafe_allow_html=True)
    
    # Text area for user input
    user_query = st.text_area("Enter your query:", placeholder="E.g., Give first three merchant names and their emails, ratings, cuisine type and reviews.", key=f"query_{len(st.session_state.interaction_history)}", value=st.session_state.get('user_query', ''))
    
    if st.button("Run Query", key=f"run_query_{len(st.session_state.interaction_history)}"):
        if user_query:
            with st.spinner("Running query..."):
                try:
                    # Define company details and agent role
                    company_details = """
                     If possible, Please always try to give answers in a table format or point wise.
                    """

                    # Prepend company details to the user's query
                    full_query = f"{company_details}\n\nUser Query: {user_query}"

                    # Execute the query using the agent
                    result = st.session_state.agent_executor.invoke(full_query)
                    st.session_state.raw_output = result['output'] if isinstance(result, dict) else result
                    
                    # Process raw output using an extraction agent 
                    extractor_llm = LLM(model="groq/llama-3.3-70b-versatile", api_key=st.session_state.api_key)
                    extractor_agent = Agent(
                        role="Data Extractor",
                        goal="Extract merchants, emails, reviews and anything possible from the raw output if they are only available.",
                        backstory="You are an expert in extracting structured information from text.",
                        provider="Groq",
                        llm=extractor_llm 
                    )
                    
                    extract_task = Task(
                        description=f"Extract a list of 'merchants' and their 'emails', 'reviews' from the following text:\n\n{st.session_state.raw_output}",
                        agent=extractor_agent,
                        expected_output="if available, Please return A structured list of merchant names, their associated email addresses, reviews etc extracted from the given text"
                    )
                    
                    # Crew execution for extraction 
                    extraction_crew = Crew(agents=[extractor_agent], tasks=[extract_task], process=Process.sequential)
                    extraction_results = extraction_crew.kickoff()
                    st.session_state.extraction_results = extraction_results if extraction_results else ""
                    st.session_state.merchant_data = st.session_state.extraction_results
                    
                    # Append the query and results to the interaction history
                    st.session_state.interaction_history.append({
                        "type": "query",
                        "content": {
                            "query": user_query,
                            "raw_output": st.session_state.raw_output,
                            "extraction_results": st.session_state.extraction_results
                        }
                    })
                    
                    # Trigger a re-run to update the UI
                    st.session_state.trigger_rerun = True
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a query before clicking 'Run Query'.")

# Display Interaction History
if st.session_state.interaction_history:
    st.markdown("### Interaction History:", unsafe_allow_html=True)
    for idx, interaction in enumerate(st.session_state.interaction_history):
        if interaction["type"] == "query":
            st.markdown(f"#### Query: {interaction['content']['query']}")
            st.markdown("**Raw Output:**")
            st.write(interaction['content']['raw_output'])
            
            # Only display extracted merchants if there is data and it does not contain ''
            if interaction['content']['extraction_results'] and interaction['content']['extraction_results'].raw and 'errorhappened' not in interaction['content']['extraction_results'].raw:
                st.markdown("**Extracted Merchants:**")
                st.write(interaction['content']['extraction_results'].raw)
        
        st.markdown("---")

# Always render the "Ask questions about your database" section
render_query_section()

# Trigger a re-run if needed
if st.session_state.trigger_rerun:
    st.session_state.trigger_rerun = False  # Reset the trigger
    st.rerun()  # Force a re-run of the script

# Footer Section 
st.markdown("---")
st.markdown(
    "<div style='text-align: center; font-size: 14px;'>Powered by <strong>Pulse iD</strong> | Built with 🐍 Python and Streamlit</div>",
    unsafe_allow_html=True 
)
