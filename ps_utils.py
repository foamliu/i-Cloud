import argparse
import logging
import os

import cv2 as cv
import numpy as np
import torch

from ps_config import MIN_MATCH_COUNT
import shapely
from shapely.geometry import Polygon, MultiPoint

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)


def ensure_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def draw_bboxes(img, points):
    for p in points:
        cv.circle(img, (int(p[0][0]), int(p[0][1])), 3, (0, 255, 0), -1)

    return img


def draw_bboxes2(img, points, color='r', thick=10):
    if color == 'g':
        rgb = (255, 0, 0)
    elif color == 'b':
        rgb = (0, 255, 0)
    else:
        rgb = (0, 0, 255)
    cv.circle(img, (int(points[0]), int(points[1])), thick, rgb, thick)
    cv.circle(img, (int(points[2]), int(points[3])), thick, rgb, thick)
    cv.circle(img, (int(points[4]), int(points[5])), thick, rgb, thick)
    cv.circle(img, (int(points[6]), int(points[7])), thick, rgb, thick)
    return img


def do_match(file1, file2):
    img1 = cv.imread(file1, 0)
    img2 = cv.imread(file2, 0)
    assert (img1.shape == img2.shape)

    h, w = img1.shape[:2]

    # print('img1.shape: ' + str(img1.shape))
    # print('img2.shape: ' + str(img1.shape))
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # print('len(good): ' + str(len(good)))

    result = None

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        # print('H: ' + str(H))
        src = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        src = np.array(src, dtype=np.float32).reshape((-1, 1, 2))
        dst = cv.perspectiveTransform(src, H)
        result = dst.tolist()
        # print('dst.shape: ' + str(dst.shape))
        # print('dst: ' + str(dst))

        # img = cv.imread(file2)
        # img = draw_bboxes(img, dst)
        # cv.imshow('', img)
        # cv.waitKey(0)

    return result


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, is_best):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    # filename = 'checkpoint_' + str(epoch) + '_' + str(loss) + '.tar'
    filename = 'checkpoint.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_checkpoint.tar')


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossMeterBag(object):
    def __init__(self, name_list):
        self.meter_dict = dict()
        self.name_list = name_list
        for name in self.name_list:
            self.meter_dict[name] = AverageMeter()

    def update(self, val_list):
        for i, name in enumerate(self.name_list):
            val = val_list[i]
            self.meter_dict[name].update(val)

    def __str__(self):
        ret = ''
        for name in self.name_list:
            ret += '{0}:\t {1:.4f}({2:.4f})\t'.format(name, self.meter_dict[name].val, self.meter_dict[name].avg)

        return ret


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--end-epoch', type=int, default=50, help='training epoch size.')
    parser.add_argument('--lr', type=float, default=0.002, help='start learning rate')
    parser.add_argument('--lr-step', type=int, default=10, help='period of learning rate decay')
    parser.add_argument('--optimizer', default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size in each context')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint')
    parser.add_argument('--use-se', type=bool, default=False, help='use SEBlock')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained model')
    args = parser.parse_args()
    return args


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


def sort_four_dot(output):
    dot1 = [output[0], output[1]]
    dot2 = [output[2], output[3]]
    dot3 = [output[4], output[5]]
    dot4 = [output[6], output[7]]
    dotstemp = {0: dot1, 1: dot2, 2: dot3, 3: dot4}
    dots = {0: dot1, 1: dot2, 2: dot3, 3: dot4}
    sum1 = output[0] + output[1]
    sum2 = output[2] + output[3]
    sum3 = output[4] + output[5]
    sum4 = output[6] + output[7]
    sums = [sum1, sum2, sum3, sum4]
    indexs = [-1, -1, -1, -1]
    # print(sums)
    # print(np.argmax(sums))
    # print(np.argmin(sums))
    mindot = np.argmin(sums)
    maxdot = np.argmax(sums)
    indexs[0] = mindot
    indexs[2] = maxdot
    dots.pop(mindot)
    dots.pop(maxdot)
    dottemp1 = dots.popitem()
    dottemp2 = dots.popitem()
    # print(dottemp1[1][0])
    # print(dottemp2)
    if dottemp1[1][0] < dottemp2[1][0]:
        indexs[1] = dottemp2[0]
        indexs[3] = dottemp1[0]
    else:
        indexs[1] = dottemp1[0]
        indexs[3] = dottemp2[0]
    # print(indexs)
    result = []
    for idx in indexs:
        result.append(dotstemp[idx][0])
        result.append(dotstemp[idx][1])
    # print(result)
    return result


def cut_and_adjust_img(img, srcdots, wide=224, height=224):
    # height1 = int((srcdots[7] - srcdots[1]) / 3)
    # height2 = int((srcdots[5] - srcdots[3]) / 3)
    height1 = 0
    height2 = 0
    src = np.array([[srcdots[0], srcdots[1]], [srcdots[2], srcdots[3]], [srcdots[4], srcdots[5] - height2], [srcdots[6], srcdots[7] - height1]], np.float32)
    dst = np.array([[0, 0], [wide, 0], [wide, height], [0, height]], np.float32)
    M3 = cv.getPerspectiveTransform(src, dst)  # 计算投影矩阵
    # print(M3)
    img2 = cv.warpPerspective(img, M3, (wide, height), borderValue=0)
    return img2


def get_iou(square1, square2):
    # square1  square2 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    a = np.array(square1).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(a).convex_hull)  # 可以打印看看是不是这样子

    b = np.array(square2).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


if __name__ == '__main__':
    square1 = [1, 0, 2, 1, 1, 2, 0, 1]
    square2 = [1, 1, 2, 0, 3, 1, 2, 2]
    print(get_iou(square1, square2))
