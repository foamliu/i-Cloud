import json

import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/matches/recognize'

    files = {'file': open('test_image_3.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])

    files = {'file': open('test_image_4.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])

    files = {'file': open('test_image_5.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])

    files = {'file': open('test_image_6.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])

    files = {'file': open('test_image_7.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])
