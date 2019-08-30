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
