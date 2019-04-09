import os
import time

from flask import request
from werkzeug.utils import secure_filename

from app.utils import compare, ensure_folder, resize


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
