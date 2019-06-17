import os
import time

import cv2 as cv
import torch
from PIL import Image
from flask import request
from torchvision import transforms
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR
from config import device
from mtcnn.detector import detect_faces
from utils import ensure_folder, draw_bboxes, crop_image, transformer

im_size = 224

checkpoint = 'BEST_checkpoint.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()


def face_attributes():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join(UPLOAD_DIR, fn)
    file.save(full_path)
    # resize(full_path)
    print('full_path: ' + full_path)

    img = Image.open(full_path).convert('RGB')
    bboxes, landmarks = detect_faces(img)

    bbox = bboxes[0]
    img = cv.imread(full_path)
    img = crop_image(img, bbox)
    img = cv.resize(img, (im_size, im_size))
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)

    inputs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
    inputs[0] = img

    with torch.no_grad():
        output = model(inputs)

    out = output.cpu().numpy()
    age_out = out[0, 0]
    pitch_out = out[0, 1]
    roll_out = out[0, 2]
    yaw_out = out[0, 3]
    beauty_out = out[0, 4]

    elapsed = time.time() - start

    return (age_out, pitch_out, roll_out, yaw_out, beauty_out), float(elapsed), str(fn)
