import os
import time

import cv2 as cv
from PIL import Image
from flask import request
from werkzeug.utils import secure_filename

from mtcnn.detector import detect_faces
from utils import compare, ensure_folder, resize, draw_bboxes, search


def face_verify():
    start = time.time()
    ensure_folder('static')
    file1 = request.files['file1']
    fn_1 = secure_filename(file1.filename)
    full_path_1 = os.path.join('static', fn_1)
    file1.save(full_path_1)
    resize(full_path_1)
    file2 = request.files['file2']
    fn_2 = secure_filename(file2.filename)
    full_path_2 = os.path.join('static', fn_2)
    file2.save(full_path_2)
    resize(full_path_2)

    prob, is_same = compare(full_path_1, full_path_2)
    elapsed = time.time() - start

    return is_same, prob, elapsed, fn_1, fn_2


def face_detect():
    start = time.time()
    ensure_folder('static')
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join('static', fn)
    file.save(full_path)
    resize(full_path)
    print('full_path: ' + full_path)

    img = Image.open(full_path).convert('RGB')
    bboxes, landmarks = detect_faces(img)
    num_faces = len(bboxes)

    if num_faces > 0:
        img = cv.imread(full_path)
        draw_bboxes(img, bboxes, landmarks)
        cv.imwrite(full_path, img)

    elapsed = time.time() - start

    return num_faces, float(elapsed), str(fn)


def face_search():
    start = time.time()
    ensure_folder('static')
    file = request.files['file']
    fn = secure_filename(file.filename)
    file_upload = os.path.join('static', fn)
    file.save(file_upload)
    resize(file_upload)
    print('file_upload: ' + file_upload)
    name, prob, file_star = search(file_upload)
    elapsed = time.time() - start
    return name, prob, file_star, file_upload, float(elapsed)
