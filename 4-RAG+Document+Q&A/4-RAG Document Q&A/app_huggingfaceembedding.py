import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv
load_dotenv()

# Load environment variables
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# use the below Huggingface embedding
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Instantiate embeddings and the model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
""")

def create_vector_embedding():
    """Function to create the vector embeddings if not initialized."""
    if "vectors" not in st.session_state:
        st.session_state["embeddings"] = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state["loader"] = PyPDFDirectoryLoader("research_papers")  # Data ingestion step
        st.session_state["docs"] = st.session_state["loader"].load()  # Load documents

        # Check if documents are loaded
        if not st.session_state["docs"]:
            st.error("No documents found in the 'research_papers' directory.")
            return

        # Splitting the documents for embedding
        st.session_state["text_splitter"] = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state["final_documents"] = st.session_state["text_splitter"].split_documents(
            st.session_state["docs"][:50]
        )
        st.session_state["vectors"] = FAISS.from_documents(
            st.session_state["final_documents"], st.session_state["embeddings"]
        )
        st.write("Vector embeddings successfully created.")
    else:
        st.write("Vector embeddings already initialized.")

# UI for the application
st.title("RAG Document Q&A With Groq And Llama3")

# User input for query
user_prompt = st.text_input("Enter your query from the research paper")

# Button to trigger document embedding
if st.button("Document Embedding"):
    create_vector_embedding()

# Ensure vectors are initialized before processing queries
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("Please initialize the document embeddings by clicking 'Document Embedding'.")
    else:
        # Create document chain and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state["vectors"].as_retriever()  # Use the embeddings to create the retriever
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Track response time
        start = time.time()
        response = retrieval_chain.invoke({'input': user_prompt})
        elapsed_time = time.time() - start

        st.write(f"Response time: {elapsed_time:.2f} seconds")
        st.write(response['answer'])

        # Display similar documents (if any)
        with st.expander("Document Similarity Search"):
            if "context" in response:
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("------------------------")
            else:
                st.write("No similar documents found.")




