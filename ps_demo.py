import pickle
import random

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

import ps_models as models
from ps_config import im_size, pickle_file, num_train, device
from ps_data_gen import data_transforms
from ps_utils import ensure_folder, draw_bboxes2, draw_bboxes,cut_and_adjust_img,sort_four_dot
import os


img_num = 32
transformer = data_transforms['valid']
checkpoint = 'ps_BEST_checkpoint_20190826.tar'
checkpoint = torch.load(checkpoint, map_location='cpu')
model = checkpoint['model']
model = model.to(device)
model.eval()

def visual_img(model):
    transformer = data_transforms['valid']

    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)

    samples = [item for item in data]
    samples = random.sample(samples, img_num)
    imgs = torch.zeros([img_num, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('images')

    for i in range(img_num):
        sample = samples[i]
        fullpath = sample['fullpath']
        raw = cv.imread(fullpath)
        raw = cv.resize(raw, (im_size, im_size))
        img = raw[..., ::-1]  # RGB
        img = transforms.ToPILImage()(img)
        img = transformer(img)
        imgs[i] = img

        cv.imwrite('images/{}_img.jpg'.format(i), raw)
        # print(sample['pts'])
        raw = draw_bboxes2(raw, sample['pts'])
        cv.imwrite('images/{}_true.jpg'.format(i), raw)

    with torch.no_grad():
        outputs = model(imgs.to(device))

    for i in range(img_num):
        output = outputs[i].cpu().numpy()
        output = output * im_size
        # print('output: ' + str(output))
        # print('output.shape: ' + str(output.shape))

        img = cv.imread('images/{}_img.jpg'.format(i))
        # print(output)
        img = draw_bboxes2(img, output)
        cv.imwrite('images/{}_out.jpg'.format(i), img)


def visual_img1(model):
    transformer = data_transforms['valid']

    files = os.listdir('real_screen')
    samples = random.sample(files, img_num)
    imgs = torch.zeros([img_num, 3, im_size, im_size], dtype=torch.float)
    ensure_folder('real_screen/images')

    for i in range(img_num):
        sample = samples[i]
        fullpath = os.path.join('real_screen', sample)
        if not 'images' in fullpath:
            print(fullpath)
            raw = cv.imread(fullpath)
            raw = cv.resize(raw, (im_size, im_size))
            img = raw[..., ::-1]  # RGB
            img = transforms.ToPILImage()(img)
            img = transformer(img)
            imgs[i] = img

            cv.imwrite('real_screen/images/{}_img.jpg'.format(i), raw)

    for i in range(img_num):
        with torch.no_grad():
            output = model(torch.unsqueeze(imgs[i], 0).to(device))
        print(output)
        output = torch.squeeze(output)     
        print(output)
        output = output.cpu().numpy()
        output = output * im_size
        # print('output: ' + str(output))
        # print('output.shape: ' + str(output.shape))

        img = cv.imread('real_screen/images/{}_img.jpg'.format(i))
        # print(output)
        img = draw_bboxes2(img, output)
        cv.imwrite('real_screen/images/{}_out.jpg'.format(i), img)

def cut_img(model, imgpath, fullpath):
    img_origin = cv.imread(imgpath)
    print(img_origin.shape)
    w, h, c = img_origin.shape
    img = cv.resize(img_origin, (im_size, im_size))
    img = img[..., ::-1]  # RGB
    img = transforms.ToPILImage()(img)
    img = transformer(img)

    with torch.no_grad():
        output = model(torch.unsqueeze(img, 0).to(device))
    print(output)
    output=output.reshape(4,-1)
    print(output)
    output = output.cpu().numpy()
    output = output * [h, w]
    output=output.reshape(-1)
    output = sort_four_dot(output)
    img = draw_bboxes2(img_origin, output)
    cv.imwrite('{}_out.jpg'.format(fullpath), img)
    img2 = cut_and_adjust_img(img, output, wide=500, height=300)
    cv.imwrite('{}_adjust.jpg'.format(fullpath), img2)

