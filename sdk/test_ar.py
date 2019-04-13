import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/ar/objects'

    obj = {'location': {'latitude': 1.0, 'longitude': 1.0, 'altitude': 1.0},
           'rotate': {'x': 0.0, 'y': 0.0, 'z': 0.0},
           'scale': {'x': 1.0, 'y': 1.0, 'z': 1.0}}
    r = requests.post(url, json=obj)
    print(r.status_code)
    print(r.text)
    obj = r.json()
    print(obj)

    r = requests.get(url)
    print(r.status_code)
    print(r.text)

    obj['location']['altitude'] = 10.0
    r = requests.put(url, data=obj)
    print(r.status_code)
    print(r.text)

    r = requests.get(url)
    print(r.status_code)
    print(r.text)

    r = requests.delete(url, json={'id': 0})
    print(r.status_code)
    print(r.text)

    r = requests.get(url)
    print(r.status_code)
    print(r.text)
