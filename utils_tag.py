import json

import redis
from flask import request

from utils import md5_hash

r = redis.Redis(host='localhost', port=6379, db=0)

en2zh = {'boy': '男', 'girl': '女', 'age0114': '0-14', 'age1519': '15-19', 'age2024': '20-24', 'age2529': '25-29',
         'age3034': '30-34', 'age3539': '35-39', 'age4044': '40-44', 'age4549': '45-49', 'age5054': '50-54',
         'age5559': '55-59', 'yf': '有房', 'yc': '有车', 'cjfh': '超级富豪', 'fh': '富豪', 'zc': '中产', 'gxjc': '工薪阶层',
         'car_bmw': '宝马', 'car_ford': '福特', 'car_kdlke': '凯迪拉克', 'car_xflan': '雪佛兰', 'car_benz': '奔驰', 'car_daz': '大众',
         'car_audi': '奥迪', 'car_gqcq': '广汽传祺', 'car_jili': '吉利', 'car_lkssi': '雷克萨斯', 'intr_fangchan': '房产',
         'intr_fdcoffee': '食品餐饮-咖啡', 'intr_fdwine': '食品餐饮-酒水', 'intr_muyin': '母婴', 'intr0124bd_jiafang': '家装百货-家具家纺',
         'intr0124fd_main': '食品餐饮-主食', 'intr0124fd_snack': '食品餐饮-小吃', 'intr0124fd_takeaway': '食品餐饮-外卖',
         'intr0124fd_tee': '食品餐饮-茶类', 'intr0125bd_jiancai': '家装百货-家居建材', 'bd_jiancai': '家装百货-家居建材',
         'intr0125ele_bigele': '家用电器-大家电', 'intr0125ele_cookele': '家用电器-厨用电器',
         'intr0125fd_importfood': '食品餐饮-进口食物', 'intr0125wear_manwear': '服装服饰-男装', 'intr0128car_midd': '汽车-中档车',
         'intr0128ele_AIfurnish': '家用电器-智能家居', 'intr0128ele_hmcomputer': '家用电器-家用电脑',
         'intr0130_ele_bath': '家用电器-卫浴家电', 'intr0130_ele_health': '家用电器-健康净化家电', 'intr0130_wear_womendress': '服装服饰-女装',
         'intr0130_wear_womenshose': '服装服饰-女鞋', 'intr0131_car_high': '汽车-高档车', 'intr0131_ele_little': '家用电器-生活小家电',
         'intr0131_wear_jewelry': '服装服饰-珠宝配饰', 'intr0131_wear_menshoes': '服装服饰-男鞋', 'intr0201_edu_middle': '教育培训-初高中教育',
         'intr0202_edu_junior': '教育培训-小学教育', 'intr0201_car_secondhand': '汽车-二手车', 'intr0201_wear_bag': '服装服饰-箱包',
         'intr0202_ca_len': '汽车-租车', 'intr0202_dig_mobi': '数码科技-手机及配件', 'intr0202_edu_bflear': '教育培训-学前教育',
         'intr0202_edu_ot': '教育培训-出国教育', 'intr0214_ca_rih': '汽车-豪华', 'intr0214_dig_mdl': '数码科技-航模车模',
         'intr0214_dig_pto': '数码科技-摄影摄像'}


def translate(en_list):
    zh_list = []
    for item in en_list:
        if item in en2zh:
            zh_list.append(en2zh[item])
    return '、'.join()


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
            gender = translate(ret['gender'])
        if 'age' in ret:
            age = translate(ret['age'])
        if 'zcdj' in ret:
            zcdj = translate(ret['zcdj'])
        if 'yf' in ret:
            yf = translate(ret['yf'])
        if 'yc' in ret:
            yc = translate(ret['yc'])
        if 'intr' in ret:
            intr = translate(ret['intr'])

    return gender, age, zcdj, yf, yc, intr
