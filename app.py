from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from opentitan_rag import OpenTitanRAG

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the RAG system
docs_dir = "raptor_output"  # Change if needed
rag = OpenTitanRAG(docs_dir=docs_dir)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '')
    use_translation = data.get('use_translation', False)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        results = rag.process_query(
            query=query,
            use_query_translation=use_translation,
            top_k=5,
            top_sources=7
        )

        response = {
            "answer": results["answer"]
        }

        # Include sub-queries if translation was used
        if use_translation and "sub_queries" in results:
            response["sub_queries"] = results["sub_queries"]
            response["retrieval_stats"] = results["retrieval_stats"]

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
