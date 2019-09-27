import os
import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from flask import request
from torchvision import transforms
from werkzeug.utils import secure_filename

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import STATIC_DIR, UPLOAD_DIR, device, logger
from models import FaceExpressionModel
from mtcnn.detector import detect_faces
from utils.common import ensure_folder, select_central_face
from utils.common import transformer

im_size = 112
# class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
class_names = ['愤怒', '厌恶', '恐惧', '高兴', '悲伤', '惊讶', '无表情']

checkpoint = 'repo/face/facial_expression.pt'
logger.info('loading model: {}...'.format(checkpoint))
model = FaceExpressionModel()
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (im_size, im_size)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (im_size, im_size)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_central_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)

    if len(landmarks) > 0:
        i = select_central_face(img.size, bounding_boxes)
        return True, [bounding_boxes[i]], [landmarks[i]]


def face_expression():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join(UPLOAD_DIR, fn)
    file.save(full_path)
    # resize(full_path)
    print('full_path: ' + full_path)

    filename = full_path
    emotion = ''
    has_face, bboxes, landmarks = get_central_face_attributes(filename)
    if has_face:
        img = align_face(filename, landmarks)
        img = img[..., ::-1]
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = torch.unsqueeze(img, dim=0)
        img = img.to(device)

        with torch.no_grad():
            pred = model(img)[0]

        pred = pred.cpu().numpy()
        pred = np.argmax(pred)
        emotion = class_names[pred]
        print(emotion)
    elapsed = time.time() - start

    return has_face, emotion, float(elapsed), full_path
