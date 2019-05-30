import requests
import json

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/matches/recognize'

    files = {'file': open('9707366-1.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
    ret = json.loads(r.text)
    print(ret['name'])
