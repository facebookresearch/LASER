#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os
import socket
import tempfile
from pathlib import Path
import numpy as np
from LASER.source.lib.text_processing import BPEfastApply
from LASER.source.embed import *

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route("/")
def root():
    print("/")
    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("LASER", "world"), hostname=socket.gethostname())


@app.route("/vectorize")
def vectorize():
    content = request.args.get('q')
    lang = request.args.get('lang')
    embedding = ''
    if lang == None or len(lang) == 0:
        lang = "en"
    # encoder
    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(model_dir, "LASER")
    model_dir = os.path.join(model_dir, "models")
    encoder_path = os.path.join(model_dir, "bilstm.93langs.2018-12-26.pt")
    bpe_codes_path = os.path.join(model_dir, "93langs.fcodes")
    print(' - Encoder: loading {}'.format(encoder_path))
    encoder = SentenceEncoder(encoder_path,
                              max_sentences=None,
                              max_tokens=12000,
                              sort_kind='mergesort',
                              cpu=True)
    with os.makedirs('./tmp') as tmpdir:
        ifname = content
        bpe_fname = os.path.join(tmpdir, 'bpe')
        bpe_oname = os.path.join(tmpdir, 'out.raw')
        print(' - BPEfastApply: bpe {}'.format(bpe_fname))
        print(' - BPEfastApply: out {}'.format(bpe_oname))
        BPEfastApply(ifname,
                     bpe_fname,
                     bpe_codes_path,
                     verbose=True, over_write=False)
        ifname = bpe_fname
        EncodeFile(encoder,
                   ifname,
                   bpe_oname,
                   verbose=True,
                   over_write=False,
                   buffer_size=10000)
        dim = 1024
        X = np.fromfile(bpe_oname.name, dtype=np.float32, count=-1)
        X.resize(X.shape[0] // dim, dim)
        embedding = X
        os.remove(bpe_fname)
        os.remove(bpe_oname)
        os.rmdir(tmpdir)
    print(lang)
    print(content)
    print(embedding)
    body = {'content': content, 'embedding': embedding, 'lang': lang}
    return jsonify(body)


if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')
