import os
import time

import cv2 as cv
import torch
from PIL import Image
from flask import request
from scipy.stats import norm
from torchvision import transforms
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR
from config import device
from mtcnn.detector import detect_faces
from utils import ensure_folder, crop_image, transformer, select_central_face, draw_bboxes

im_size = 224

checkpoint = 'models/attributes/BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()


def get_prob(beauty):
    mu = 49.1982
    sigma = 14.0220
    prob = norm.cdf(beauty, mu, sigma)
    return prob


expression_dict = {0: '无表情', 1: '微笑', 2: '大笑'}
face_shape_dict = {0: 'square', 1: 'oval', 2: 'heart', 3: 'round', 4: 'triangle'}
face_type_dict = {0: 'human', 1: 'cartoon'}
gender_dict = {0: '女', 1: '男'}
glasses_dict = {0: '无眼镜', 1: '太阳镜', 2: '一般眼镜'}
race_dict = {0: '黄色', 1: '白色', 2: '黑色', 3: '阿拉伯人'}


def idx2name(idx, tag):
    name = None
    if tag == 'expression':
        name = expression_dict[idx]
    elif tag == 'face_shape':
        name = face_shape_dict[idx]
    elif tag == 'face_type':
        name = face_type_dict[idx]
    elif tag == 'gender':
        name = gender_dict[idx]
    elif tag == 'glasses':
        name = glasses_dict[idx]
    elif tag == 'race':
        name = race_dict[idx]
    return name


def name2idx(name):
    lookup_table = {'none': 0, 'smile': 1, 'laugh': 2,
                    'square': 0, 'oval': 1, 'heart': 2, 'round': 3, 'triangle': 4,
                    'human': 0, 'cartoon': 1,
                    'female': 0, 'male': 1,
                    'sun': 1, 'common': 2,
                    'yellow': 0, 'white': 1, 'black': 2, 'arabs': 3}

    return lookup_table[name]


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

    result = None

    if len(bboxes) > 0:
        i = select_central_face((im_size, im_size), bboxes)
        bbox = bboxes[i]
        img = cv.imread(full_path)
        boxed = draw_bboxes(img, [bbox], [landmarks[i]])
        cv.imwrite(full_path, boxed)
        img = crop_image(img, bbox)
        img = cv.resize(img, (im_size, im_size))
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        img = img.to(device)

        inputs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
        inputs[0] = img

        with torch.no_grad():
            reg_out, expression_out, gender_out, glasses_out, race_out = model(inputs)

        reg_out = reg_out.cpu().numpy()
        age_out = reg_out[0, 0]
        pitch_out = reg_out[0, 1]
        roll_out = reg_out[0, 2]
        yaw_out = reg_out[0, 3]
        beauty_out = reg_out[0, 4]

        age = int(age_out * 100)
        pitch = float('{0:.2f}'.format(pitch_out * 360 - 180))
        roll = float('{0:.2f}'.format(roll_out * 360 - 180))
        yaw = float('{0:.2f}'.format(yaw_out * 360 - 180))
        beauty = float('{0:.2f}'.format(beauty_out * 100))
        beauty_prob = float('{0:.4f}'.format(get_prob(beauty)))

        _, expression_out = expression_out.topk(1, 1, True, True)
        _, gender_out = gender_out.topk(1, 1, True, True)
        _, glasses_out = glasses_out.topk(1, 1, True, True)
        _, race_out = race_out.topk(1, 1, True, True)
        expression_out = expression_out.cpu().numpy()
        gender_out = gender_out.cpu().numpy()
        glasses_out = glasses_out.cpu().numpy()
        race_out = race_out.cpu().numpy()

        expression = idx2name(int(expression_out[i][0]), 'expression')
        gender = idx2name(int(gender_out[i][0]), 'gender')
        glasses = idx2name(int(glasses_out[i][0]), 'glasses')
        race = idx2name(int(race_out[i][0]), 'race')

        result = {'age': age, 'pitch': pitch, 'roll': roll, 'yaw': yaw, 'beauty': beauty, 'beauty_prob': beauty_prob,
                  'expression': expression, 'gender': gender, 'glasses': glasses, 'race': race}

    elapsed = time.time() - start

    return result, float(elapsed), str(fn)
