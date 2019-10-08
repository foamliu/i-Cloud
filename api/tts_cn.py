from flask import jsonify

from utils.asr import do_recognize
from . import api


@api.route('/asr/recognize', methods=['POST'])
def asr_recognize():
    text, file_upload, elapsed = do_recognize()
    return jsonify({'text': text, 'file_upload': file_upload, 'elapsed': elapsed})
