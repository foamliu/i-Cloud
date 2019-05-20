from flask import request


def search_tag():
    mac = request.form.get('macAddress')
    gender = mac
    age = ''
    zcdj = ''
    intr = ''
    return gender, age, zcdj, intr
