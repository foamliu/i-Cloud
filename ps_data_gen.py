import pickle
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms

from ps_config import im_size, pickle_file, num_train

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class FrameDetectionDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        samples = [item for item in data]
        num = len(samples)
        num_test = int(num / 10)
        num_train = num - num_test
        if split == 'train':
            self.samples = samples[:num_train]
        else:
            self.samples = samples[num_train:]

        self.transformer = data_transforms[split]

    def __getitem__(self, i):
        sample = self.samples[i]
        fullpath = sample['fullpath']
        img = cv.imread(fullpath)
        img = cv.resize(img, (im_size, im_size))
        img = img[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = self.transformer(img)

        pts = np.array(sample['pts'])
        pts = pts.reshape((8,))
        pts = pts / im_size
        pts = np.clip(pts, 0, 1)

        return img, pts

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = FrameDetectionDataset('train')
    print(dataset[0])
    print(len(dataset))
