# src/api/rag_api.py

import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add the project root to the Python path
# This allows us to import our RAGQueryHandler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.rag_module.core import RAGQueryHandler

# --- Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for our frontend

# Initialize our RAG Query Handler
# This is a global instance that will be created once when the API starts.
# This is efficient as it avoids reloading the model and DB on every request.
try:
    rag_handler = RAGQueryHandler()
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    print("Cannot start the RAG API without the vector database.")
    # In a real production app, you might exit here or handle it differently
    rag_handler = None

# --- API Endpoints ---
@app.route('/api/query', methods=['POST'])
def handle_query():
    """
    Handles POST requests to the /api/query endpoint.
    Expects a JSON payload with a 'question' key.
    """
    if rag_handler is None:
        return jsonify({"error": "RAG handler is not initialized. Database may be missing."}), 500

    if not request.is_json:
        return jsonify({"error": "Invalid input: request must be JSON."}), 400

    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    try:
        # Use our handler to get the relevant documents
        retrieved_docs = rag_handler.query(question)

        # Format the response to be clean and useful for the frontend
        response_data = []
        for doc in retrieved_docs:
            response_data.append({
                "source": os.path.basename(doc.metadata.get("source", "Unknown")),
                "page_content": doc.page_content
            })

        return jsonify({"answer_chunks": response_data})

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # This block runs when the script is executed directly for testing
    # It starts the Flask development server
    app.run(host='0.0.0.0', port=5001, debug=True)