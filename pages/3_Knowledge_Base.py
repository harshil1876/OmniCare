import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Set up the embedding model
embeddings = HuggingFaceEmbeddings()

# Function to handle file upload and processing
def setup_knowledge_base(uploaded_file):
    if uploaded_file is not None:
        # Read the file content
        file_contents = uploaded_file.read().decode("utf-8")
        
        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(file_contents)
        
        # Create a Chroma vector store
        db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")
        db.persist()
        st.success("Knowledge base created successfully!")
        return db
    return None

st.header("Knowledge Base")
st.write("Upload a document to create a searchable knowledge base.")

uploaded_file = st.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    if st.button("Create Knowledge Base"):
        setup_knowledge_base(uploaded_file)

st.subheader("Search the Knowledge Base")
query = st.text_input("Enter your search query:")

if query:
    # Load the persisted database
    if os.path.exists("./chroma_db"):
        db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
        docs = db.similarity_search(query)
        if docs:
            st.write("Found relevant documents:")
            for doc in docs:
                st.write(doc.page_content)
        else:
            st.write("No relevant documents found.")
    else:
        st.warning("Please create a knowledge base first.")
