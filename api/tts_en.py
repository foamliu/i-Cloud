import time

from flask import jsonify

from utils.tts_en import synthesize_en
from . import api


@api.route('/asr/synthesize_en', methods=['GET'])
def do_synthesize_en():
    start = time.time()
    text = 'hello 123456789'
    filename = synthesize_en(text)
    elapsed = time.time() - start
    elapsed = float(elapsed)
    return jsonify({'text': text, 'filename': filename, 'elapsed': elapsed})
