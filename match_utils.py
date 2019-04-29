import os
import time

from flask import request
from werkzeug.utils import secure_filename

from utils import compare, ensure_folder, resize


def face_verify():
    start = time.time()
    ensure_folder('static')
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join('static', fn)
    file.save(full_path)
    resize(full_path)

    prob, is_same = compare(full_path)
    elapsed = time.time() - start

    return is_same, prob, elapsed
