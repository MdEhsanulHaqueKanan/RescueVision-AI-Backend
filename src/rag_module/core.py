# src/rag_module/core.py

import os
import sys

# Add the project root to the Python path
# This allows us to import modules from the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- Define constants for the paths ---
# This makes the code cleaner and easier to maintain
ROOT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
VECTOR_DB_PATH = os.path.join(ROOT_DIRECTORY, "vector_store")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGQueryHandler:
    """
    Handles the logic for querying the RAG knowledge base.
    """
    def __init__(self):
        """
        Initializes the query handler by loading the embedding model and vector database.
        This is done once to be efficient.
        """
        print("--- Initializing RAGQueryHandler ---")

        # Check if the vector database path exists
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"Vector database not found at '{VECTOR_DB_PATH}'. "
                f"Please run 'python scripts/build_vector_db.py' to create it."
            )

        # Load the embedding model
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")

        # Load the persisted Chroma database
        self.db = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
        print("Vector database loaded successfully.")
        print("--- RAGQueryHandler is ready. ---")

    def query(self, query_text: str, k: int = 3) -> list:
        """
        Performs a similarity search on the vector database.

        Args:
            query_text (str): The user's question.
            k (int): The number of relevant chunks to retrieve.

        Returns:
            list: A list of Document objects containing the relevant chunks.
        """
        print(f"\nPerforming similarity search for query: '{query_text}'")
        retrieved_docs = self.db.similarity_search(query_text, k=k)
        print(f"Found {len(retrieved_docs)} relevant documents.")
        return retrieved_docs

# --- Example of how to use this module ---
if __name__ == '__main__':
    # This block runs when the script is executed directly for testing
    print("--- Testing RAGQueryHandler Module ---")
    
    # Initialize the handler
    handler = RAGQueryHandler()

    # Define a sample query
    sample_query = "What are the responsibilities of a SAR team leader?"

    # Perform the query
    results = handler.query(sample_query)

    # Print the results
    print("\n--- Top 3 most relevant chunks found: ---")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} (Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}) ---")
        print(doc.page_content)