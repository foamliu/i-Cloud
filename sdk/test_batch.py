import json

import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/faces/get_feature_batch'
    print(url)
    files = {'file': open('test_image_1.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    d = json.loads(r.text)
    print(d['feature'])