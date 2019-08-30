import os
import time

import torch
from flask import request
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR
from config import device
from utils import ensure_folder

checkpoint = 'models/asr-cn/BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()


def do_recognize():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename.lower()
    file_upload = os.path.join(UPLOAD_DIR, filename)
    file.save(file_upload)
    print('file_upload: ' + file_upload)
    text = ''
    elapsed = time.time() - start
    elapsed = float(elapsed)
    return text, file_upload, elapsed
