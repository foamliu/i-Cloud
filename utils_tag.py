import redis
from flask import request

from utils import md5_hash

r = redis.Redis(host='localhost', port=6379, db=0)


def search_tag():
    mac = str(request.form.get('macAddress'))
    mac = mac.replace(':', '')
    md5 = md5_hash(mac)
    ret = r.get(md5)
    gender = age = zcdj = yf = yc = intr = None
    if 'gender' in ret:
        gender = ret['gender']
    if 'age' in ret:
        age = ret['age']
    if 'zcdj' in ret:
        zcdj = ret['zcdj']
    if 'yf' in ret:
        yf = ret['yf']
    if 'yc' in ret:
        yc = ret['yc']
    if 'intr' in ret:
        intr = ret['intr']

    return gender, age, zcdj, yf, yc, intr
