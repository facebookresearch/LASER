#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import socket

from flask import Flask, jsonify, request

from laser_encoders import LaserEncoderPipeline
from laser_encoders.language_list import LASER2_LANGUAGE, LASER3_LANGUAGE

app = Flask(__name__)

# Global cache for encoders
encoder_cache = {}

laser2_encoder = None


@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("LASER", "world"), hostname=socket.gethostname())


@app.route("/vectorize", methods=["GET"])
def vectorize():
    content = request.args.get("q")
    lang = request.args.get(
        "lang", "eng"
    )  # Default to English if 'lang' is not provided

    if content is None:
        return jsonify({"error": "Missing input content"}), 400

    try:
        global laser2_encoder
        if lang in LASER2_LANGUAGE:  # Checks for both 3-letter code or 8-letter code
            if not laser2_encoder:
                laser2_encoder = LaserEncoderPipeline(lang=lang)
            encoder = laser2_encoder
        else:
            lang_code = LASER3_LANGUAGE.get(
                lang, lang
            )  # Use language code as key to prevent multiple entries for same language
            if lang_code not in encoder_cache:
                encoder_cache[lang_code] = LaserEncoderPipeline(lang=lang_code)
            encoder = encoder_cache[lang_code]

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
