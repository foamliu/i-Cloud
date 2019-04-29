import math
import os
import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from flask import request
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
threshold = 50
fps = 25.

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


def video_match():
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
    print(max_index)
    print(max_value)

    theta = math.acos(max_value)
    theta = theta * 180 / math.pi

    time_in_video = 1 / fps * max_index
    elapsed = time.time() - start

    return theta < threshold, max_index, time_in_video, elapsed, fn
