import streamlit as st
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import os

# Initialize the LLM
llm = OllamaLLM(model="llama2:latest")

# Function to create and run the agent
def run_agent(query):
    # Check if the sales data file exists
    if not os.path.exists("data/sales_data.csv"):
        st.error("sales_data.csv not found in /data folder!")
        st.stop()
        
    # Load the sales data
    df = pd.read_csv("data/sales_data.csv")
    
    # Create the pandas agent
    pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
    
    # Improved query routing
    sales_keywords = ["sales", "revenue", "profit", "data"]
    is_sales_query = any(keyword in query.lower() for keyword in sales_keywords)
    
    # Load the ChromaDB knowledge base
    if os.path.exists("./chroma_db"):
        embeddings = HuggingFaceEmbeddings()
        vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        
        if is_sales_query:
            response = pandas_agent.invoke(query, handle_parsing_errors=True)
        else:
            docs = vector_store.similarity_search(query)
            if docs:
                response = "\n".join([doc.page_content for doc in docs])
            else:
                response = "I couldn't find any relevant information in the knowledge base."
    else:
        if is_sales_query:
            response = pandas_agent.invoke(query, handle_parsing_errors=True)
        else:
            response = "No knowledge base available. Please upload a document in the Knowledge Base page to create one."
        
    return response

st.header("AI Agents")
st.write("Interact with the AI agents to get insights from your data and knowledge base.")

query = st.text_input("Ask the AI agent a question:")

if query:
    with st.spinner("The AI agent is thinking..."):
        response = run_agent(query)
        st.write(response)
