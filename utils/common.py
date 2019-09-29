import hashlib
import os

from flask import request
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def md5_hash(mac):
    mac = mac.upper()
    mac = ':'.join(mac[i:i + 2] for i in range(0, 12, 2))
    mac = mac.encode('utf-8')
    m = hashlib.md5()
    m.update(mac)
    md5 = m.hexdigest()
    return md5


def normalize_mac(mac):
    mac = mac.upper()
    mac = ':'.join(mac[i:i + 2] for i in range(0, 12, 2))
    return mac


def crop_image(img, bbox):
    # print(bbox.shape)
    height, width = img.shape[:2]
    x1 = int(round(bbox[0]))
    x1 = max(0, x1)
    y1 = int(round(bbox[1]))
    y1 = max(0, y1)
    x2 = int(round(bbox[2]))
    x2 = min(width - 1, x2)
    y2 = int(round(bbox[3]))
    y2 = min(height - 1, y2)
    # w = int(abs(x2 - x1))
    # h = int(abs(y2 - y1))
    # print(x1, y1, w, h)
    # print(img.shape)
    # print('x1:{} y1:{} w:{} h:{}'.format(x1, y1, w, h))
    crop_img = img[y1:y2, x1:x2]
    return crop_img


def save_file():
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join(UPLOAD_DIR, fn)
    file.save(full_path)
    # resize(full_path)
    print('full_path: ' + full_path)
    return full_path
