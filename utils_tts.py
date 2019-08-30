import time

from flask import request


def do_synthesize():
    start = time.time()
    text = request.form['text']
    print('text: ' + str(text))
    elapsed = time.time() - start
    elapsed = float(elapsed)
    audiopath = ''
    return audiopath, elapsed
