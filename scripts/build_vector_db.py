# scripts/build_vector_db.py

import os
import sys

# Add the project root to the Python path
# This allows us to import modules from the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Define the paths relative to the script location
# __file__ is the path to the current script
# os.path.dirname(__file__) is the directory the script is in ('scripts')
# os.path.join(..., '..') goes one level up to the project root
ROOT_DIRECTORY = os.path.join(os.path.dirname(__file__), '..')
KNOWLEDGE_BASE_DIR = os.path.join(ROOT_DIRECTORY, "data", "knowledge_base")
VECTOR_DB_PATH = os.path.join(ROOT_DIRECTORY, "vector_store")

def build_vector_database():
    """
    Builds the vector database from PDF documents in the knowledge base.
    """
    print("--- Starting to build the vector database ---")

    # 1. Load documents
    print(f"Loading documents from: {KNOWLEDGE_BASE_DIR}")
    loader = PyPDFDirectoryLoader(KNOWLEDGE_BASE_DIR)
    documents = loader.load()
    if not documents:
        print("No documents found. Please check the knowledge_base directory.")
        return
    print(f"Successfully loaded {len(documents)} pages.")

    # 2. Chunk documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split documents into {len(chunks)} chunks.")

    # 3. Create embeddings and store in ChromaDB
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embedding model initialized.")

    print(f"Creating and persisting vector database at: {VECTOR_DB_PATH}")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"--- Successfully built and saved the vector database with {vectordb._collection.count()} entries. ---")

if __name__ == "__main__":
    # This block runs when the script is executed directly
    build_vector_database()