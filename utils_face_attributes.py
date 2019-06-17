import os
import time
from scipy.stats import norm
import cv2 as cv
import torch
from PIL import Image
from flask import request
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
    prob = norm.pdf(beauty, mu, sigma)
    return prob


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
            output = model(inputs)

        out = output.cpu().numpy()
        age_out = out[0, 0]
        pitch_out = out[0, 1]
        roll_out = out[0, 2]
        yaw_out = out[0, 3]
        beauty_out = out[0, 4]

        age = int(age_out * 100)
        pitch = float('{0:.2f}'.format(pitch_out * 360 - 180))
        roll = float('{0:.2f}'.format(roll_out * 360 - 180))
        yaw = float('{0:.2f}'.format(yaw_out * 360 - 180))
        beauty = float('{0:.2f}'.format(beauty_out * 100))
        beauty_prob = float('{0:.4f}'.format(get_prob(beauty)))

        result = {'age': age, 'pitch': pitch, 'roll': roll, 'yaw': yaw, 'beauty': beauty, 'beauty_prob': beauty_prob}

    elapsed = time.time() - start

    return result, float(elapsed), str(fn)
