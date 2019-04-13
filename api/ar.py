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
    return make_response(jsonify({'id': next_id}), 201)


@api.route('/ar/objects', methods=['GET'])
def get():
    return jsonify(list(ID2OBJ.values()))


@api.route('/ar/objects/<int:id>', methods=['GET', 'PUT', 'DELETE'])
def process(id, **kwargs):
    if request.method == 'DELETE':
        if id in ID2OBJ:
            del ID2OBJ[id]
        return {
                   "message": "Object {} deleted successfully".format(id)
               }, 200
    elif request.method == 'PUT':
        obj = request.data
        print(obj)
        response = jsonify({
            "message": "Object {} updated successfully".format(id)
        })
        response.status_code = 200
        return response
    else:
        # GET
        response = jsonify(ID2OBJ[id])
        response.status_code = 200
        return response
