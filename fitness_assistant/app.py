"""Flask API exposing the RAG pipeline."""

from __future__ import annotations

import uuid
from typing import Any, Dict

from flask import Flask, jsonify, request

from fitness_assistant.rag import rag

app = Flask(__name__)


@app.post("/question")
def ask_question():
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    question = payload.get("question")
    size = payload.get("size")
    skip_llm = bool(payload.get("skip_llm", False))

    if not question or not isinstance(question, str):
        return jsonify({"error": "Missing or invalid 'question' field."}), 400

    try:
        result = rag(question, size=size, skip_llm=skip_llm)
    except Exception as exc:  # pragma: no cover - surfaced to client
        return jsonify({"error": str(exc)}), 500

    conversation_id = str(uuid.uuid4())
    response = {
        "conversation_id": conversation_id,
        "question": question
        # "context": result.get("context", ""),
        # "documents": result.get("documents", []),
    }
    if "answer" in result:
        response["answer"] = result["answer"]
    return jsonify(response), 200


@app.post("/feedback")
def submit_feedback():
    payload: Dict[str, Any] = request.get_json(force=True, silent=False) or {}
    conversation_id = payload.get("conversation_id")
    feedback = payload.get("feedback")

    if not conversation_id or not isinstance(conversation_id, str):
        return jsonify({"error": "Missing or invalid 'conversation_id' field."}), 400

    if feedback not in (-1, 1):
        return jsonify({"error": "Feedback must be either +1 or -1."}), 400

    return jsonify(
        {
            "conversation_id": conversation_id,
            "feedback": feedback,
            "status": "received",
        }
    ), 200


if __name__ == "__main__":  # pragma: no cover
    app.run(host="0.0.0.0", port=5000, debug=False)
