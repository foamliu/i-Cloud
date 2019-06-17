import hashlib
import math
import pickle

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from scipy.stats import norm
from torchvision import transforms

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import image_h, image_w, device
from mtcnn.detector import detect_faces

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

checkpoint = 'BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

# model params
threshold = 73.18799151798612
mu_0 = 89.6058
sigma_0 = 4.5451
mu_1 = 43.5357
sigma_1 = 8.83


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


def get_image(filename):
    has_face, bboxes, landmarks = get_central_face_attributes(filename)
    if not has_face:
        raise FaceNotFoundError(filename)

    img = align_face(filename, landmarks)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)

    print('drawing bboxes: {}'.format(filename))
    bboxes, landmarks = get_all_face_attributes(filename)
    pic = cv.imread(filename)
    pic = draw_bboxes(pic, bboxes, landmarks)
    cv.imwrite(filename, pic)

    return img


def resize(filename):
    img = cv.imread(filename)
    h, w = img.shape[:2]
    ratio_w = w / 1280
    ratio_h = h / 720
    if ratio_w > 1 or ratio_h > 1:
        ratio = max(ratio_w, ratio_h)
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        img = cv.resize(img, (new_w, new_h))
        cv.imwrite(filename, img)


def compare(fn_0, fn_1):
    print('fn_0: ' + fn_0)
    print('fn_1: ' + fn_1)
    img0 = get_image(fn_0)
    img1 = get_image(fn_1)
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = img0
    imgs[1] = img1

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


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def md5_hash(mac):
    mac = mac.upper()
    mac = ':'.join(mac[i:i + 2] for i in range(0, 12, 2))
    mac = mac.encode('utf-8')
    m = hashlib.md5()
    m.update(mac)
    md5 = m.hexdigest()
    return md5


def normalize_mac(mac):
    mac = mac.upper()
    mac = ':'.join(mac[i:i + 2] for i in range(0, 12, 2))
    return mac


def crop_image(img, bbox):
    # print(bbox.shape)
    x1 = int(round(bbox[0]))
    y1 = int(round(bbox[1]))
    x2 = int(round(bbox[2]))
    y2 = int(round(bbox[3]))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    # print(x1, y1, w, h)
    # print(img.shape)
    # print('x1:{} y1:{} w:{} h:{}'.format(x1, y1, w, h))
    crop_img = img[y1:y1 + h, x1:x1 + w]
    return crop_img


if __name__ == "__main__":
    compare('id_card.jpg', 'photo_1.jpg')
    compare('id_card.jpg', 'photo_2.jpg')
    compare('id_card.jpg', 'photo_3.jpg')
    compare('id_card.jpg', 'photo_4.jpg')
