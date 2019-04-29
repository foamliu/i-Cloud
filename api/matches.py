from flask import jsonify

from utils_match import video_match
from . import api


@api.route('/matches/recognize', methods=['POST'])
def recognize():
    is_match, index, time_in_video, elapsed, fn = video_match()
    return jsonify({'is_match': is_match, 'index': index, 'time_in_video': time_in_video, 'elapsed': elapsed})
