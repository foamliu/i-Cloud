import requests
import json

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/asr/recognize'
    print(url)
    text = 'hello 123456789'
    json = {'text': text}
    r = requests.post(url, json=json)
    print(r.status_code)
    print(r.text)
    data = json.loads(r.text)
    filename = data['filename']
    elapsed = data['elapsed']
    print('filename: ' + filename)
    print('elapsed: ' + str(elapsed))
