from flask import jsonify

from utils_face import face_verify, face_detect, face_search
from . import api


@api.route('/faces/verify', methods=['POST'])
def verify():
    is_same, prob, elapsed, fn_1, fn_2 = face_verify()
    return jsonify({'is_same': is_same, 'prob': prob, 'elapsed': elapsed})


@api.route('/faces/detect', methods=['POST'])
def detect():
    num_faces, elapsed, fn, _ = face_detect()
    return jsonify({'num_faces': num_faces, 'elapsed': elapsed})


@api.route('/faces/search', methods=['POST'])
def search():
    name, prob, file_star, file_upload, elapsed = face_search()
    return jsonify({'name': name, 'prob': prob, 'elapsed': elapsed, 'file_star': file_star, 'file_upload': file_upload})
