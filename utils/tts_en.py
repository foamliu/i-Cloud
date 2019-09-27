import os
import time

import numpy as np
import soundfile as sf
import torch
from flask import request

from config import device, logger
from tacotron2.tacotron2 import Tacotron2
from text import symbols
from text import text_to_sequence


class HParams:
    def __init__(self):
        self.n_mel_channels = 80
        self.dynamic_loss_scaling = True
        self.fp16_run = False
        self.distributed_run = False

        ################################
        # Data Parameters             #
        ################################
        self.load_mel_from_disk = False

        ################################
        # Audio Parameters             #
        ################################
        self.max_wav_value = 32768.0
        self.sampling_rate = 22050
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = 80
        self.mel_fmin = 0.0
        self.mel_fmax = 8000.0

        ################################
        # Model Parameters             #
        ################################
        self.n_symbols = len(symbols)
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 1  # currently only 1 is supported
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.batch_size = 64
        self.mask_padding = True  # set model's padded outputs to padded values


config = HParams()
checkpoint = 'repo/tts-en/tacotron2-en.pt'
logger.info('loading model: {}...'.format(checkpoint))
model = Tacotron2(config)
model.load_state_dict(torch.load(checkpoint))
model = model.to(device)
model.eval()

# waveglow_path = 'waveglow_256channels.pt'
# waveglow = torch.load(waveglow_path)['model']
# waveglow.cuda().eval().half()
# for k in waveglow.convinv:
#     k.float()
from utils.tts_cn import waveglow

sampling_rate = 22050


def do_synthesize_en():
    start = time.time()
    text = request.form['text']
    print('text: ' + str(text))
    audiopath = synthesize_en(text)
    elapsed = time.time() - start
    elapsed = float(elapsed)
    return audiopath, elapsed


def synthesize_en(text):
    print(text)
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    mel_outputs_postnet = mel_outputs_postnet.type(torch.float16)
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)

    audio = audio[0].data.cpu().numpy()
    audio = audio.astype(np.float32)

    print('audio.shape: ' + str(audio.shape))
    print(audio)
    filename = time.strftime("%Y%m%d-%H%M%S") + '.wav'
    sf.write(os.path.join('static', filename), audio, sampling_rate, 'PCM_24')
    return filename
