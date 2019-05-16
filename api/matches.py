from flask import jsonify

from utils_match import match_video
from . import api


@api.route('/matches/recognize', methods=['POST'])
def recognize():
    is_match, prob, index, time_in_video, elapsed, fn = match_video()
    return jsonify(
        {'is_match': is_match, 'index': index, 'time_in_video': time_in_video, 'elapsed': elapsed})
