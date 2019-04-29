from flask import jsonify

from utils_face import face_verify
from . import api


@api.route('/faces/verify', methods=['POST'])
def verify():
    is_same, prob, elapsed, fn_1, fn_2 = face_verify()
    return jsonify({'is_same': is_same, 'prob': prob, 'elapsed': elapsed})
