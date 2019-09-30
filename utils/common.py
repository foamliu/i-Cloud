import hashlib
import os
import random

import cv2 as cv
from flask import request
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR, logger


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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
    logger.info('request received')
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    rand = random.randint(10000, 99999)
    filename = '{}_{}.{}'.format(name, rand, ext)
    full_path = os.path.join(UPLOAD_DIR, filename)
    file.save(full_path)
    # resize(full_path)
    logger.info('file transferred, full_path: ' + full_path)
    return full_path


def resize(filename):
    img = cv.imread(filename)
    h, w = img.shape[:2]
    ratio_w = w / 1280
    ratio_h = h / 720
    if ratio_w > 1 or ratio_h > 1:
        ratio = max(ratio_w, ratio_h)
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        img = cv.resize(img, (new_w, new_h))
        cv.imwrite(filename, img)
