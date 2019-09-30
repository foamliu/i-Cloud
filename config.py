import logging
import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
image_w = 112
image_h = 112

num_classes = 85164

STATIC_DIR = 'static'
UPLOAD_FOLDER = 'upload'
UPLOAD_DIR = os.path.join(STATIC_DIR, UPLOAD_FOLDER)
pickle_file = STATIC_DIR + '/' + 'weiweiimage.pkl'

IGNORE_ID = -1


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    # formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)]s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
