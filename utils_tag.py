from flask import request

def search_tag():
    mac = request.form['macAddress']
    gender = mac
    age = ''
    zcdj = ''
    intr = ''
    return gender, age, zcdj, intr
