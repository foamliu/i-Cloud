import math
import os
import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from flask import request
from scipy.stats import norm
from torchvision import transforms
from werkzeug.utils import secure_filename

from config import device, im_size
from utils import ensure_folder, resize

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
transformer = data_transforms['val']

checkpoint = 'models/match/BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# model params
mat = np.load('static/video.npy')


def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def gen_feature(filename):
    img = cv.imread(filename)
    img = cv.resize(img, (im_size, im_size))
    imgs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
    imgs[0] = get_image(img, transformer)
    features = model(imgs.to(device)).cpu().numpy()
    feature = features[0]
    feature = feature / np.linalg.norm(feature)
    return feature


def match_video():
    start = time.time()
    ensure_folder('static')
    file = request.files['file']
    fn = secure_filename(file.filename)
    full_path = os.path.join('static', fn)
    file.save(full_path)
    resize(full_path)
    print('full_path: ' + full_path)

    with torch.no_grad():
        feature = gen_feature(full_path)

    cosine = np.dot(mat, feature)
    cosine = np.clip(cosine, -1, 1)
    print(cosine.shape)
    max_index = np.argmax(cosine)
    max_value = cosine[max_index]

    theta = math.acos(max_value)
    theta = theta * 180 / math.pi
    print('theta: ' + str(theta))

    fps = 25.
    time_in_video = 1 / fps * max_index
    elapsed = time.time() - start

    threshold = 43.12986168973048
    prob = get_prob(theta)
    return theta < threshold, prob, int(max_index), float(time_in_video), float(elapsed), str(fn)


def compare(full_path_1, full_path_2):
    with torch.no_grad():
        feature_1 = gen_feature(full_path_1)
        feature_2 = gen_feature(full_path_2)

    x0 = feature_1 / np.linalg.norm(feature_1)
    x1 = feature_2 / np.linalg.norm(feature_2)
    cosine = np.dot(x0, x1)
    cosine = np.clip(cosine, -1, 1)
    theta = math.acos(cosine)
    theta = theta * 180 / math.pi

    threshold = 43.12986168973048
    is_match = theta < threshold
    prob = get_prob(theta)
    return is_match, prob


def get_prob(theta):
    mu_0 = 88.7952
    sigma_0 = 15.5666
    mu_1 = 12.5701
    sigma_1 = 7.22
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    total = prob_0 + prob_1
    return prob_1 / total


def match_image():
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

    is_match = compare(full_path_1, full_path_2)
    elapsed = time.time() - start

    return is_match, elapsed, fn_1, fn_2
