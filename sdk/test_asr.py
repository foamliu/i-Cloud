import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/asr/recognize'
    print(url)
    files = {'file': open('audio_5.wav', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
