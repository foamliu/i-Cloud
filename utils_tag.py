import json

import redis
from flask import request

from utils import md5_hash

r = redis.Redis(host='localhost', port=6379, db=0)


def search_tag():
    gender = age = zcdj = yf = yc = intr = None
    mac = str(request.form.get('macAddress'))
    mac = mac.replace(':', '')
    md5 = md5_hash(mac)
    ret = r.get(md5)
    if ret:
        ret = ret.decode()
        ret = json.loads(ret)
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
