from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import uuid
from opentitan_rag import OpenTitanRAG

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For secure sessions
CORS(app, supports_credentials=True)  # Enable CORS with credentials

# Initialize the RAG system
docs_dir = "raptor_output"  # Change if needed
# Base URL for source links
base_url = "https://github.com/lowRISC/opentitan/blob/master/"
rag = OpenTitanRAG(docs_dir=docs_dir, base_url=base_url)

# Store session_ids for users
sessions = {}


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    use_translation = data.get('use_translation', False)

    # Get or create session_id
    session_id = data.get('session_id')

    # If no session_id provided or invalid, create a new one
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {'messages': []}

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = rag.process_query(
            query=query,
            session_id=session_id,  # Pass session_id to maintain chat history
            use_query_translation=use_translation,
            top_k=5,
            top_sources=7
        )

        # Store messages in session
        sessions[session_id]['messages'].append({
            'query': query,
            'answer': results["answer"]
        })

        response = {
            "answer": results["answer"],
            "session_id": session_id  # Return session_id to client
        }

        # Include sub-queries if translation was used
        if use_translation and "sub_queries" in results:
            response["sub_queries"] = results["sub_queries"]
            response["retrieval_stats"] = results["retrieval_stats"]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear-chat', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id')

    if session_id and session_id in sessions:
        # Clear session data
        sessions[session_id]['messages'] = []
        # Clear RAG chat history
        rag.clear_chat_history(session_id)
        return jsonify({"status": "success", "message": "Chat history cleared"})

    return jsonify({"status": "success", "message": "Session not found, created new session"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)
