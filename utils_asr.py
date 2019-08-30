import os
import time

import librosa
import numpy as np
import torch
from flask import request
from werkzeug.utils import secure_filename

from config import STATIC_DIR, UPLOAD_DIR
from config import device
from utils import ensure_folder

checkpoint = 'models/asr-cn/BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

input_dim = 80
LFR_m = 4
LFR_n = 3


def extract_feature(input_file, feature='fbank', dim=40, cmvn=True, delta=False, delta_delta=False,
                    window_size=25, stride=10, save_feature=None):
    y, sr = librosa.load(input_file, sr=None)
    ws = int(sr * 0.001 * window_size)
    st = int(sr * 0.001 * stride)
    if feature == 'fbank':  # log-scaled
        feat = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=dim,
                                              n_fft=ws, hop_length=st)
        feat = np.log(feat + 1e-6)
    elif feature == 'mfcc':
        feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=dim, n_mels=26,
                                    n_fft=ws, hop_length=st)
        feat[0] = librosa.feature.rmse(y, hop_length=st, frame_length=ws)

    else:
        raise ValueError('Unsupported Acoustic Feature: ' + feature)

    feat = [feat]
    if delta:
        feat.append(librosa.feature.delta(feat[0]))

    if delta_delta:
        feat.append(librosa.feature.delta(feat[0], order=2))
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[:, np.newaxis]
    if save_feature is not None:
        tmp = np.swapaxes(feat, 0, 1).astype('float32')
        np.save(save_feature, tmp)
        return len(tmp)
    else:
        return np.swapaxes(feat, 0, 1).astype('float32')


def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.
    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)


def do_recognize():
    start = time.time()
    ensure_folder(STATIC_DIR)
    ensure_folder(UPLOAD_DIR)
    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename.lower()
    file_upload = os.path.join(UPLOAD_DIR, filename)
    file.save(file_upload)
    print('file_upload: ' + file_upload)
    text = recognize(file_upload)
    elapsed = time.time() - start
    elapsed = float(elapsed)
    return text, file_upload, elapsed


def recognize(audiopath):
    feature = extract_feature(input_file=audiopath, feature='fbank', dim=input_dim, cmvn=True)
    feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
    # feature = np.expand_dims(feature, axis=0)
    input = torch.from_numpy(feature).to(device)
    input_length = [input[0].shape[0]]
    input_length = torch.LongTensor(input_length).to(device)
    nbest_hyps = model.recognize(input, input_length, char_list, args)
    out_list = []
    for hyp in nbest_hyps:
        out = hyp['yseq']
        out = [char_list[idx] for idx in out]
        out = ''.join(out)
        out_list.append(out)
    print('OUT_LIST: {}'.format(out_list))

    return out_list
