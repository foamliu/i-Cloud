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
import pickle
from config import device
from utils import ensure_folder, resize

im_size = 224

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
pickle_file = 'data/video_index.pkl'

threshold = 25.50393648495902
mu_0 = 46.1028
sigma_0 = 6.4981
mu_1 = 9.6851
sigma_1 = 3.060

with open(pickle_file, 'rb') as file:
    frames = pickle.load(file)

num_frames = len(frames)
features = np.empty((num_frames, 512), dtype=np.float32)
name_list = []
idx_list = []
fps_list = []

for i, frame in enumerate(frames):
    name = frame['name']
    feature = frame['feature']
    fps = frame['fps']
    idx = frame['idx']
    features[i] = feature
    name_list.append(name)
    idx_list.append(idx)
    fps_list.append(fps)

print(features.shape)
assert (len(name_list) == num_frames)


def get_image(img):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def gen_feature(filename):
    img = cv.imread(filename)
    img = cv.resize(img, (im_size, im_size))
    imgs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
    imgs[0] = get_image(img)
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
        x = gen_feature(full_path)

    cosine = np.dot(features, x)
    cosine = np.clip(cosine, -1, 1)
    print('cosine.shape: ' + str(cosine.shape))
    max_index = int(np.argmax(cosine))
    max_value = cosine[max_index]
    name = name_list[max_index]
    fps = fps_list[max_index]
    idx = idx_list[max_index]
    print('max_index: ' + str(max_index))
    print('max_value: ' + str(max_value))
    print('name: ' + name)
    print('fps: ' + str(fps))
    print('idx: ' + str(idx))
    theta = math.acos(max_value)
    theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
    time_in_video = idx / fps
    print('time_in_video: ' + str(time_in_video))

    prob = get_prob(theta)
    elapsed = time.time() - start
    return name, prob, idx, float(time_in_video), float(elapsed), str(fn)


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

    is_match = theta < threshold
    prob = get_prob(theta)
    return is_match, prob


def get_prob(theta):
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
