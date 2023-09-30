#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import socket
from flask import Flask, jsonify, request
from laser_encoders import initialize_encoder, initialize_tokenizer
import numpy as np

app = Flask(__name__)


@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" \
            "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("LASER", "world"), hostname=socket.gethostname())

@app.route("/vectorize", methods=["GET"])
def vectorize():
    content = request.args.get('q')
    lang = request.args.get('lang', 'en')  # Default to English if 'lang' is not provided

    if content is None:
        return jsonify({'error': 'Missing input content'})

    encoder = initialize_encoder(lang=lang)
    tokenizer = initialize_tokenizer(lang=lang)

    # Tokenize the input content
    tokenized_sentence = tokenizer.tokenize(content)

    # Encode the tokenized sentence
    embeddings = encoder.encode_sentences([tokenized_sentence])
    embeddings_list = embeddings.tolist()

    body = {'content': content, 'embedding': embeddings_list}
    return jsonify(body)

if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')

