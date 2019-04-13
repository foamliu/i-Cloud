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


@api.route('/ar/objects', methods=['DELETE'])
def delete():
    id_dict = request.get_json()
    id = id_dict['id']
    if id in ID2OBJ:
        del ID2OBJ[id]
    return make_response(jsonify(message='Object deleted'), 200)
