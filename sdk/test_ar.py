import json

import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/ar/objects'

    # POST
    obj = {'location': {'latitude': 1.0, 'longitude': 1.0, 'altitude': 1.0},
           'rotate': {'x': 0.0, 'y': 0.0, 'z': 0.0},
           'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0}}
    r = requests.post(url, json=obj)
    print('POST response:')
    print(r.status_code)
    print(r.text)

    id = r.json()['id']
    url_with_id = 'http://47.101.196.204:8080/api/v1/ar/objects/{}'.format(id)

    # GET
    r = requests.get(url_with_id)
    print('GET response:')
    print(r.status_code)
    print(r.text)

    # PUT
    obj['location']['altitude'] = 10.0
    r = requests.put(url_with_id, data=json.dumps(obj))
    print('PUT response:')
    print(r.status_code)
    print(r.text)

    # GET
    r = requests.get(url_with_id)
    print('GET response:')
    print(r.status_code)
    print(r.text)

    # DELETE
    r = requests.delete(url_with_id)
    print('DELETE response:')
    print(r.status_code)
    print(r.text)

    # GET LIST
    r = requests.get(url)
    print('GET LIST response:')
    print(r.status_code)
    print(r.text)
