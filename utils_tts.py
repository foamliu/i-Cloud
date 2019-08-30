import json
import time

import numpy as np
import pinyin
import soundfile as sf
import torch
from flask import request

from config import device

checkpoint = 'repo/tts-cn/BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

sampling_rate = 22050

vocab_file = 'models/tts-cn/vocab.json'

with open(vocab_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

VOCAB = data['VOCAB']
IVOCAB = data['IVOCAB']


def text_to_sequence(text):
    result = [VOCAB[ch] for ch in text]
    return result


def do_synthesize():
    start = time.time()
    text = request.form['text']
    print('text: ' + str(text))
    elapsed = time.time() - start
    elapsed = float(elapsed)
    audiopath = synthesize(text)
    return audiopath, elapsed


def synthesize(text):
    waveglow_path = 'waveglow_256channels.pt'
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    # text = "相对论直接和间接的催生了量子力学的诞生 也为研究微观世界的高速运动确立了全新的数学模型"
    text = pinyin.get(text, format="numerical", delimiter=" ")
    print(text)
    sequence = np.array(text_to_sequence(text))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel_outputs_postnet = mel_outputs_postnet.type(torch.float16)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = audio.astype(np.float32)

    print('audio.shape: ' + str(audio.shape))
    print(audio)

    sf.write('static/output.wav', audio, sampling_rate, 'PCM_24')
    return 'output.wav'
