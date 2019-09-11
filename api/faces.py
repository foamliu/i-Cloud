from flask import jsonify

from utils_face import face_verify, face_detect, face_search, face_feature
from utils_face_attributes import face_attributes
from . import api


@api.route('/faces/verify', methods=['POST'])
def verify():
    is_same, prob, elapsed, fn_1, fn_2 = face_verify()
    return jsonify({'is_same': is_same, 'prob': prob, 'elapsed': elapsed})


@api.route('/faces/detect', methods=['POST'])
def detect():
    try:
        num_faces, elapsed, fn, bboxes, landmarks = face_detect()
        return jsonify(
            {'num_faces': num_faces, 'bboxes': bboxes.tolist(), 'landmarks': landmarks.tolist(), 'elapsed': elapsed})
    except Exception as err:
        print(err)
        return jsonify({'num_faces': 0})


@api.route('/faces/search', methods=['POST'])
def search():
    name, prob, file_star, file_upload, elapsed = face_search()
    file_star = file_star.replace('data', '')
    file_upload = file_upload.replace('static', '')
    return jsonify({'name': name, 'prob': prob, 'elapsed': elapsed, 'file_star': file_star, 'file_upload': file_upload})


@api.route('/faces/attributes', methods=['POST'])
def attributes():
    result, elapsed, file_upload = face_attributes()
    file_upload = file_upload.replace('static', '')
    return jsonify({'result': result, 'elapsed': elapsed, 'file_upload': file_upload})


@api.route('/faces/get_feature', methods=['POST'])
def get_feature():
    feature, file_upload, elapsed = face_feature()
    file_upload = file_upload.replace('static', '')
    return jsonify({'feature': feature.tolist(), 'elapsed': elapsed, 'file_upload': file_upload})
