import math
import os
import pickle
import random
import time
import zipfile

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from flask import request
from scipy.stats import norm
from torchvision import transforms
from werkzeug.utils import secure_filename

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import STATIC_DIR, UPLOAD_DIR
from config import image_h, image_w, device, logger
from models import resnet101
from mtcnn.detector import detect_faces
from utils.common import ensure_folder, resize

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
transformer = data_transforms['val']


class HParams:
    def __init__(self):
        self.pretrained = False
        self.use_se = True


config = HParams()

checkpoint = 'repo/face/insight-face-v3.pt'
logger.info('loading model: {}...'.format(checkpoint))
model = resnet101(config)
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()

# model params
threshold = 76.75066649278368
mu_0 = 89.76046947988898
sigma_0 = 4.498024182861556
mu_1 = 42.66766813673472
sigma_1 = 8.62761102672923


class FaceNotFoundError(Exception):
    """Base class for other exceptions"""
    pass


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_central_face_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except:
        pass
    return False, None, None


def get_all_face_attributes(full_path):
    img = Image.open(full_path).convert('RGB')
    bounding_boxes, landmarks = detect_faces(img)
    return bounding_boxes, landmarks


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[1] + b[3]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def draw_bboxes(img, bounding_boxes, facial_landmarks=[]):
    for b in bounding_boxes:
        cv.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255), 1)

    for p in facial_landmarks:
        for i in range(5):
            cv.circle(img, (int(p[i]), int(p[i + 5])), 1, (0, 255, 0), -1)

        break  # only first

    return img


def get_image(filename, flip=False, draw=True):
    has_face, bboxes, landmarks = get_central_face_attributes(filename)
    if not has_face:
        raise FaceNotFoundError(filename)

    img = align_face(filename, landmarks)
    if flip:
        img = np.flip(img, 1)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)

    if draw:
        print('drawing bboxes: {}'.format(filename))
        bboxes, landmarks = get_all_face_attributes(filename)
        pic = cv.imread(filename)
        pic = draw_bboxes(pic, bboxes, landmarks)
        cv.imwrite(filename, pic)

    return img


def compare(fn_0, fn_1):
    print('fn_0: ' + fn_0)
    print('fn_1: ' + fn_1)
    img0 = get_image(fn_0)
    img1 = get_image(fn_1)
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = img0
    imgs[1] = img1
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)

        feature0 = output[0].cpu().numpy()
        feature1 = output[1].cpu().numpy()
        x0 = feature0 / np.linalg.norm(feature0)
        x1 = feature1 / np.linalg.norm(feature1)
        cosine = np.dot(x0, x1)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
    return prob, theta < threshold


def get_prob(theta):
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total


def search(full_path):
    img = get_image(full_path)
    imgs = torch.zeros([1, 3, 112, 112], dtype=torch.float)
    imgs[0] = img
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)

        feature = output[0].cpu().numpy()
        x = feature / np.linalg.norm(feature)

    with open('static/stars.pkl', 'rb') as file:
        data = pickle.load(file)

    names = data['names']
    files = data['files']
    features = data['features']

    cosine = np.dot(features, x)
    cosine = np.clip(cosine, -1, 1)
    print('cosine.shape: ' + str(cosine.shape))
    max_index = int(np.argmax(cosine))
    max_value = cosine[max_index]
    print('max_index: ' + str(max_index))
    print('max_value: ' + str(max_value))
    print('name: ' + names[max_index])
    print('file: ' + files[max_index])
    theta = math.acos(max_value)
    theta = theta * 180 / math.pi
    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))

    return names[max_index], prob, files[max_index]


def get_feature(full_path):
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = get_image(full_path, draw=False)
    imgs[1] = get_image(full_path, flip=True, draw=False)
    imgs = imgs.to(device)

    with torch.no_grad():
        output = model(imgs)

        feature = output[0].cpu().numpy() + output[1].cpu().numpy()
        x = feature / np.linalg.norm(feature)

    return x


def face_verify():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file1 = request.files['file1']
    fn_1 = secure_filename(file1.filename)
    full_path_1 = os.path.join(UPLOAD_DIR, fn_1)
    file1.save(full_path_1)
    resize(full_path_1)
    file2 = request.files['file2']
    fn_2 = secure_filename(file2.filename)
    full_path_2 = os.path.join(UPLOAD_DIR, fn_2)
    file2.save(full_path_2)
    resize(full_path_2)

    prob, is_same = compare(full_path_1, full_path_2)
    elapsed = time.time() - start

    return is_same, prob, elapsed, fn_1, fn_2


def face_detect(full_path):
    start = time.time()
    img = Image.open(full_path).convert('RGB')
    bboxes, landmarks = detect_faces(img)
    num_faces = len(bboxes)

    if num_faces > 0:
        img = cv.imread(full_path)
        draw_bboxes(img, bboxes, landmarks)
        cv.imwrite(full_path, img)

    elapsed = time.time() - start

    return num_faces, float(elapsed), bboxes, landmarks


def face_search():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename.lower()
    if filename in ['jpg', 'jpeg', 'png', 'gif']:
        filename = str(random.randint(0, 101)) + '.' + filename
    file_upload = os.path.join(UPLOAD_DIR, filename)
    file.save(file_upload)
    resize(file_upload)
    print('file_upload: ' + file_upload)
    name, prob, file_star = search(file_upload)
    elapsed = time.time() - start
    return name, prob, file_star, file_upload, float(elapsed)


def face_feature():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename.lower()
    if filename in ['jpg', 'jpeg', 'png', 'gif']:
        filename = str(random.randint(0, 101)) + '.' + filename
    file_upload = os.path.join(UPLOAD_DIR, filename)
    file.save(file_upload)
    resize(file_upload)
    print('file_upload: ' + file_upload)
    feature = get_feature(file_upload)
    elapsed = time.time() - start
    return feature, file_upload, float(elapsed)


def extract(filename, folder_path):
    print('Extracting {}...'.format(filename))
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(folder_path)
    zip_ref.close()


def face_feature_batch(full_path=''):
    start = time.time()
    folder_path = 'static/batch'
    ensure_folder(folder_path)
    if full_path.lower().endswith('.zip'):
        extract(full_path, folder_path)
    elapsed = time.time() - start
    return full_path, elapsed


if __name__ == "__main__":
    compare('id_card.jpg', 'photo_1.jpg')
    compare('id_card.jpg', 'photo_2.jpg')
    compare('id_card.jpg', 'photo_3.jpg')
    compare('id_card.jpg', 'photo_4.jpg')
