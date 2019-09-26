import torch

from config import device, logger
from models import EastModel


class HParams:
    def __init__(self):
        self.pretrained = True
        self.network = 'r50'


config = HParams()

checkpoint = 'repo/asr-cn/speech-transformer-cn.pt'
logger.info('loading model: {}...'.format(checkpoint))
model = EastModel(config)
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()


def detect():
    pass
