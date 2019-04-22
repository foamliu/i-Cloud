import requests

if __name__ == '__main__':
    url = 'http://47.101.196.204:8080/api/v1/faces/verify'

    files = {'file1': open('sample_verify_1.jpg', 'rb'), 'file2': open('sample_verify_2.jpg', 'rb')}
    r = requests.post(url, files=files)
    print(r.status_code)
    print(r.text)
