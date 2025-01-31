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
    # parser.add_argument(
    #     '--cont',
    #     type=str,
    #     default = "last",
    #     help='''Latest saved epoch checkpoint used for generation'''
    #          '''Or, specify which epoch to use.'''
    # )

    parser.add_argument(
        '--epochs',
        nargs='*',
        type=int,
        help='''Which saved epoch of model to load'''
             '''if list of epoch numbers: will generate outputs using each epoch's model'''
             '''if no argument provided: load the latest epoch'''
    )

    parser.add_argument(
        '--latent_vals',
        nargs='*',
        type=int,
        help='''Value of the latent code vectors'''
             '''(for extrapolating outside training range, ie.e [15,0,0,....])'''
             '''if list of value: will generate outputs using each value '''
             '''if no argument provided: will use default value 1'''
    )

    parser.add_argument(
        '--num_categ',
        type=int,
        default=10,
        help='Q-net categories'
        # length of latent code vector
    )
    parser.add_argument(
        '--num_examples',
        type=int,
        default=100,
        help='Number of examples to generate for each latent code'
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
    if len(args.epochs) == 0:
        epochs = ['last']
    else:
        epochs = args.epochs
    if len(args.latent_vals) == 0:
        latent_vals = [1]
    else:
        latent_vals = args.latent_vals
    log_dir = args.logdir
    out_dir_main = args.outdir
    # creates main outputs directory: ./gen_outputs
    os.makedirs(out_dir_main, exist_ok=True)
    sample_rate = args.sample_rate
    slice_len = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EXAMPLES = args.num_examples


    # Load generator from saved checkpoint, specified by --cont parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for target_epoch in epochs:
        fname, eps = get_continuation_fname(target_epoch, log_dir)
        out_dir_epoch = os.path.join(out_dir_main, f'{eps}epochs')
        # creates sub-directory: ./gen_outputs/{num_eps}epochs
        os.makedirs(out_dir_epoch, exist_ok=True)
        print(f"---- Loaded model saved from Epoch {eps} ----")
        print(f"---- Evaluating a ciwGAN with {NUM_CATEG} categories ----")
        G = WaveGANGenerator(slice_len=slice_len)
        G.load_state_dict(torch.load(os.path.join(log_dir, fname + "_G.pt"),
                                    map_location = device))
        G.to(device)
        G.eval()

        # Generate from random noise
        # Added: Manipulation of latent code vector c (for one-hot vector, ciw)
        # Added: Setting values of latent code outside training range

        class_values = torch.arange(0, NUM_CATEG)
        latent_codes = torch.nn.functional.one_hot(class_values, num_classes=NUM_CATEG).to(device)
        # replaces each class value i --> a one-hot vector [0,..1,..0] representing that class 
        
        for i in range(NUM_CATEG):
            c = torch.reshape(latent_codes[i], (1, NUM_CATEG))
            # set the value of the latent codes to values outside of training range
            for val in latent_vals:
                # separate each latent code value into different directories
                out_dir_val = os.path.join(out_dir_epoch, f"val{val}")
                os.makedirs(out_dir_val, exist_ok=True)
                c_ = c * val
                print(f"---- Saving generated outputs to {out_dir_val} ----")
                # for each latent code vector c_ (outside train range), generate NUM_EXAMPLES
                for j in range(NUM_EXAMPLES):
                    # z : input noise --> add manipulation of latent code c
                    _z = torch.FloatTensor(1, 100 - NUM_CATEG).uniform_(-1, 1).to(device)
                    z = torch.cat((c_, _z), dim=1)
                    # print(z)
                    assert z.shape == (1, 100)
                    genData = G(z)[0, 0, :].detach().cpu().numpy()
                    output = (genData * 32767).astype(np.int16)    # convert output value range
                    write(os.path.join(out_dir_val, f"code{i}-ex{j}-val{val}.wav"), sample_rate, genData)
                    
