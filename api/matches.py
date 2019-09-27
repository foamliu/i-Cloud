from flask import jsonify

from utils.common import match_video
from . import api


@api.route('/matches/recognize', methods=['POST'])
def recognize():
    name, prob, idx, time_in_video, elapsed, upload_file, image_fn = match_video()
    return jsonify(
        {'name': name, 'prob': prob, 'index': idx, 'time_in_video': time_in_video, 'elapsed': elapsed,
         'upload_file': upload_file, 'image_fn': image_fn})
