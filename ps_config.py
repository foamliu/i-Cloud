import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
im_size = 224
channel = 3

# Training parameters
num_workers = 1  # for data-loading; right now, only 1 works with h5py
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none
MIN_MATCH_COUNT = 10
loss_ratio = im_size

# Data parameters
num_samples = 10  # before filtering: 373471
num_tests = 5
num_train = num_samples - num_tests
DATA_DIR = 'data'
IMG_DIR = '../Image-Matching/data/cron20190326_resized/'

pickle_file = 'data/data.pkl'
