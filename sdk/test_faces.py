import json

import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/faces/verify'
    print(url)
    files = {'file1': open('test_image_1.jpg', 'rb'), 'file2': open('test_image_2.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)

    url = 'http://47.101.196.204:8080/api/v1/faces/detect'
    print(url)
    files = {'file': open('test_image_1.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)

    url = 'http://47.101.196.204:8080/api/v1/faces/search'
    print(url)
    files = {'file': open('test_image_1.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    d = json.loads(r.text)
    print(d['name'])

    url = 'http://47.101.196.204:8080/api/v1/faces/attributes'
    print(url)
    files = {'file': open('test_image_1.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
