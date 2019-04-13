from flask import jsonify

from . import api

objects = []


@api.route('/ar/objects', methods=['POST'])
def post():
    return jsonify(objects)


@api.route('/ar/objects', methods=['GET'])
def get():
    objects.clear()
    objects.append({'location': {'latitude': 1.0, 'longitude': 1.0, 'altitude': 1.0},
                    'rotate': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                    'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0}})
    return jsonify(objects)
