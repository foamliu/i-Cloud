import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors
image_w = 224
image_h = 224
im_size = 224
num_classes = 85164
