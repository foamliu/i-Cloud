from flask import jsonify

from utils_asr import do_recognize
from . import api


@api.route('/asr/recognize', methods=['POST'])
def recognize():
    text, file_upload, elapsed = do_recognize()
    return jsonify({'text': text, 'file_upload': file_upload, 'elapsed': elapsed})
