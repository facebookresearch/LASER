#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import socket

from flask import Flask, jsonify, request
from laser_encoders import LaserEncoderPipeline

app = Flask(__name__)

# Global cache for encoders
encoder_cache = {}

@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("LASER", "world"), hostname=socket.gethostname())

@app.route("/vectorize", methods=["GET"])
def vectorize():
    content = request.args.get("q")
    lang = request.args.get("lang", "eng")  # Default to English if 'lang' is not provided

    if content is None:
        return jsonify({"error": "Missing input content"}), 400

    try:
        # Use cached encoder if available, else create a new one
        if lang not in encoder_cache:
            encoder_cache[lang] = LaserEncoderPipeline(lang=lang)
        encoder = encoder_cache[lang]

        embeddings = encoder.encode_sentences([content])
        embeddings_list = embeddings.tolist()
        body = {"content": content, "embedding": embeddings_list}
        return jsonify(body), 200

    except ValueError as e:
        # Check if the exception is due to an unsupported language
        if "unsupported language" in str(e).lower():
            return jsonify({"error": f"Language '{lang}' is not supported."}), 400
        else:
            return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=80, host="0.0.0.0")
