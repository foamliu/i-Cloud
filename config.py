import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
image_w = 112
image_h = 112

num_classes = 85164

STATIC_DIR = 'static'
UPLOAD_DIR = os.path.join(STATIC_DIR, 'upload')
pickle_file = STATIC_DIR + '/' + 'weiweiimage.pkl'
