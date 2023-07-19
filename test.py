# -*- coding: utf-8 -*-
"""
Author: Andrej Leban
Created on Sun May 29 13:05:27 2022
"""

import argparse
import os
import time
# from train import NUM_CATEG

# import sounddevice as sd
# import soundfile as sf
import numpy as np
from scipy.io.wavfile import write
import torch

from infowavegan import WaveGANGenerator
from utils import get_continuation_fname

# cf: https://github.com/pytorch/pytorch/issues/16797
# class CPU_Unpickler(pk.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else:
#             return super().find_class(module, name)

if __name__ == "__main__":
    # generator = CPU_Unpickler(open("generator.pkl", 'rb')).load()
    # discriminator = CPU_Unpickler(open("discriminator.pkl", 'rb')).load()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Directory where checkpoints are saved'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        required=True,
        help="Directory where generated outputs are stored"
    )
    parser.add_argument(
        '--cont',
        type=str,
        default = "last",
        help='Latest saved epoch checkpoint used for generation'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        help='Latest saved epoch checkpoint used for generation'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=10,
        help='Q-net categories'
        # length of latent code vector
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Sample rate'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384
    )

    args = parser.parse_args()
    # epoch = args.epoch
    cont = args.cont
    dir = args.logdir
    out_dir = args.outdir
    sample_rate = args.sample_rate
    slice_len = args.slice_len
    NUM_CATEG = args.num_categ

    # Load generator from checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fname, eps = get_continuation_fname(cont, dir)
    print(f"---- Loaded model saved from Epoch {eps} ----")
    G = WaveGANGenerator(slice_len=slice_len)
    G.load_state_dict(torch.load(os.path.join(dir, fname + "_G.pt"),
                                 map_location = device))
    G.to(device)
    G.eval()

    # Generate from random noise
    # Added: Manipulation of latent code vector c (for one-hot vector, ciw)

    class_values = torch.arange(0, NUM_CATEG)
    latent_codes = torch.nn.functional.one_hot(class_values, num_classes=NUM_CATEG).to(device)
    for i in range(NUM_CATEG):
        c = torch.reshape(latent_codes[i], (1, NUM_CATEG))
        # for each latent code vector c, generate 20 examples 
        for j in range(20):
            # z : input noise --> add manipulation of latent code c
            _z = torch.FloatTensor(1, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
            z = torch.cat((c, _z), dim=1)
            assert z.shape == (1, 100)
            genData = G(z)[0, 0, :].detach().cpu().numpy()
            # write(f'out.wav', sample_rate, (genData * 32767).astype(np.int16))
            # sd.play(genData, sample_rate)
            # time.sleep(1)
            write(os.path.join(out_dir, f"{i}-{j}.wav"), sample_rate, genData)
