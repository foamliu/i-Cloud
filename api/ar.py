from flask import jsonify
from flask import make_response
from flask import request

from . import api

ID2OBJ = dict()


@api.route('/ar/objects', methods=['POST'])
def post():
    obj = request.get_json()
    next_id = len(ID2OBJ)
    obj['id'] = next_id
    ID2OBJ[next_id] = obj
    return make_response(jsonify(message='Object created'), 201)


@api.route('/ar/objects', methods=['GET'])
def get():
    return jsonify(list(ID2OBJ.values()))
