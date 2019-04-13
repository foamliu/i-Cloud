import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/ar/objects'
    r = requests.get(url)
    print(r.status_code)
    print(r.text)
