from flask import jsonify

from api import api
from ..face_utils import face_verify


@api.route('/faces/verify', methods=['POST'])
def verify():
    is_same, prob, elapsed, fn_1, fn_2 = face_verify()
    return jsonify({'is_same': is_same, 'prob': prob, 'elapsed': elapsed})
