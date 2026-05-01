from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "LLM service running"})

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json

    # For now: mock response
    return jsonify({
        "summary": "Mock analysis",
        "competitors": ["Competitor A", "Competitor B"],
        "marketSize": "Large",
        "opportunities": ["Growth", "Innovation"],
        "risks": ["Competition"],
        "swotAnalysis": "Mock SWOT"
    })

if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", 8000)))