import os
import time

import cv2 as cv
import torch
from PIL import Image
from flask import request
from scipy.stats import norm
from torchvision import transforms
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR, device, logger
from models import FaceAttributeModel
from mtcnn.detector import detect_faces
from utils.common import ensure_folder, crop_image, transformer, select_central_face, draw_bboxes

im_size = 224

checkpoint = 'repo/attributes/face-attributes.pt'
logger.info('loading model: {}...'.format(checkpoint))
model = FaceAttributeModel()
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()

