import argparse
import os
import re

import numpy as np
import torch
import torch.optim as optim
from scipy.io.wavfile import read
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from infowavegan import WaveGANGenerator, WaveGANDiscriminator, WaveGANQNetwork
from utils import get_continuation_fname


class AudioDataSet:
    def __init__(self, datadir, slice_len):
        print("Loading data")
        dir = os.listdir(datadir)
        x = np.zeros((len(dir), 1, slice_len))
        i = 0
        for file in tqdm(dir):
            # print("Reading file:", file, " with path:",  os.path.join(datadir, file))
            # ignoring hidden file sthat starts with ._
            if file.startswith("."):
                continue
            audio = read(os.path.join(datadir, file))[1]
            if audio.shape[0] < slice_len:
                audio = np.pad(audio, (0, slice_len - audio.shape[0]))
            audio = audio[:slice_len]

            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32767
            elif audio.dtype == np.float32:
                pass
            else:
                raise NotImplementedError('Scipy cannot process atypical WAV files.')
            audio /= np.max(np.abs(audio))
            x[i, 0, :] = audio
            i += 1

        self.len = len(x)
        self.audio = torch.from_numpy(np.array(x, dtype=np.float32))

    def __getitem__(self, index):
        return self.audio[index]

    def __len__(self):
        return self.len


def gradient_penalty(G, D, real, fake, epsilon):
    x_hat = epsilon * real + (1 - epsilon) * fake
    scores = D(x_hat)
    grad = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True
    )[0]
    grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1)  # norm along each batch
    penalty = ((grad_norm - 1) ** 2).unsqueeze(1)
    return penalty


if __name__ == "__main__":
    # Training Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datadir',
        type=str,
        required=True,
        help='Training Directory'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        required=True,
        help='Log/Results Directory'
    )
    parser.add_argument(
        '--num_categ',
        type=int,
        default=5,
        help='Q-net categories'
        # length of latent code vector c
    )
    # whether or no Q-net's optimizer updates G's parameters,
    # defaults to false
    parser.add_argument(
        '--Q_update_G',
        action='store_true',
        help='Q-net optimizer will update G parameters'
    )
    parser.set_defaults(Q_update_G=False)
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=5000,
        help='Epochs'
    )
    parser.add_argument(
        '--slice_len',
        type=int,
        default=16384,
        help='Length of training data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--cont',
        type=str,
        default="",
        help='''continue: default from the last saved iteration. '''
             '''Provide the epoch number if you wish to resume from a specific point'''
             '''Or set "last" to continue from last available'''
    )

    parser.add_argument(
        '--save_int',
        type=int,
        default=50,
        help='Save interval in epochs'
    )

    # Q-net Arguments
    Q_group = parser.add_mutually_exclusive_group()
    Q_group.add_argument(
        '--ciw',
        action='store_true',
        help='Trains a ciwgan'
    )
    Q_group.add_argument(
        '--fiw',
        action='store_true',
        help='Trains a fiwgan'
    )
    args = parser.parse_args()
    train_Q = args.ciw or args.fiw

    # Parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datadir = args.datadir
    logdir = args.logdir
    SLICE_LEN = args.slice_len
    NUM_CATEG = args.num_categ
    NUM_EPOCHS = args.num_epochs
    WAVEGAN_DISC_NUPDATES = 5
    BATCH_SIZE = args.batch_size
    LAMBDA = 10
    LEARNING_RATE = 1e-4
    BETA1 = 0.5
    BETA2 = 0.9

    CONT = args.cont
    SAVE_INT = args.save_int

    # Load data
    dataset = AudioDataSet(datadir, SLICE_LEN)
    dataloader = DataLoader(
        dataset,
        BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    def make_new():
        G = WaveGANGenerator(slice_len=SLICE_LEN, ).to(device).train()
        D = WaveGANDiscriminator(slice_len=SLICE_LEN).to(device).train()

        # Optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
        optimizer_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

        Q, optimizer_Q, criterion_Q = (None, None, None)
        if train_Q:
            Q = WaveGANQNetwork(slice_len=SLICE_LEN, num_categ=NUM_CATEG).to(device).train()
            if args.Q_update_G:
                # allow Q-network's optimizer to also update the Generator's parameters
                combined_params = list(Q.parameters()) + list(G.parameters())
                optimizer_Q = optim.RMSprop(combined_params, lr=LEARNING_RATE)
                print("Q-network optimizer will update Generator")
            else:
                optimizer_Q = optim.RMSprop(Q.parameters(), lr=LEARNING_RATE) 
                print("Q-network optimizer will NOT update Generator")
        if args.fiw:
            print("Training a fiwGAN with ", NUM_CATEG, " categories.")
            # fiw: latent code is binary bit-vector 
            criterion_Q = torch.nn.BCEWithLogitsLoss()
        elif args.ciw:
            # ciw: latent code is one-hot vector
            print("Training a ciwGAN with ", NUM_CATEG, " categories.")
            # Question: each time criterion_Q is called --> a new CrossEntropyLoss object is created
            criterion_Q = lambda inpt, target: torch.nn.CrossEntropyLoss()(inpt, target.max(dim=1)[1])

        return G, D, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q

    # Load models
    G, D, optimizer_G, optimizer_D, Q, optimizer_Q, criterion_Q = make_new()
    start_epoch = 0
    start_step = 0

    if CONT.lower() != "":
        try:
            print("Loading model from existing checkpoints...")
            fname, start_epoch = get_continuation_fname(CONT, logdir)

            G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_G.pt")))
            D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_D.pt")))
            if train_Q:
                Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Q.pt")))

            optimizer_G.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Gopt.pt")))
            optimizer_D.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Dopt.pt")))

            if train_Q:
                optimizer_Q.load_state_dict(torch.load(f=os.path.join(logdir, fname + "_Qopt.pt")))

            start_step = int(re.search(r'_step(\d+).*', fname).group(1))
            print(f"Successfully loaded model. Continuing training from epoch {start_epoch}, step {start_step}")

        # Don't care why it failed
        except Exception as e:
            print("Could not load from existing checkpoint, initializing new model...")
            print(e)
    else:
        print("Starting a new training")

    # Set Up Writer
    writer = SummaryWriter(logdir)
    step = start_step
    ### Begin Training ###
    for epoch in range(start_epoch + 1, NUM_EPOCHS):

        print("Epoch {} of {}".format(epoch, NUM_EPOCHS))
        print("-----------------------------------------")
        pbar = tqdm(dataloader)
        real = dataset[:BATCH_SIZE].to(device)

        # iterate over "real" audio recordings in dataloader, in mini-batches
        # i: i-th batch in the full dataset
        # real: mini-batch of recorded audio vectors
        # each "flush" of mini-batches through the network amounts to 1 "step"
        for i, real in enumerate(pbar):
            # D Update --> D learns to classify real vs fake data
            optimizer_D.zero_grad()
            real = real.to(device)
            epsilon = torch.rand(BATCH_SIZE, 1, 1).repeat(1, 1, SLICE_LEN).to(device)
            _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)

            if train_Q:
                if args.fiw:
                    # fiw: Binary vector, each entry is Bernoulli (0 or 1)
                    c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                else:
                    # ciw: One-hot vector, only (randomly chosen) entry is 1, all else 0
                    c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                    num_classes=NUM_CATEG).to(device)
                # Input vector to generator network = (latent code c, len=num_cate) + (random noise z)
                z = torch.cat((c, _z), dim=1)
            else:
                z = _z

            fake = G(z)
            penalty = gradient_penalty(G, D, real, fake, epsilon)

            # Value function: D(x) - D(G(z))
            # D tries to maximize --> want to predict real D(x) as positive and predict fake D(G(z)) as negative
            # G tries to minimize --> want to trick D(G(z)) to be predicted as positive, so -D(G(z)) will be negative
            D_loss = torch.mean(D(fake) - D(real) + LAMBDA * penalty)
            writer.add_scalar('Loss/Discriminator', D_loss.detach().item(), step)
            D_loss.backward()
            optimizer_D.step()

            # Updating G (only update G after 5 updates of D)
            if i % WAVEGAN_DISC_NUPDATES == 0:
                optimizer_G.zero_grad()
                if train_Q:
                    optimizer_Q.zero_grad()

                _z = torch.FloatTensor(BATCH_SIZE, 100 - NUM_CATEG).uniform_(-1, 1).to(device)

                if train_Q:
                    if args.fiw:
                        c = torch.FloatTensor(BATCH_SIZE, NUM_CATEG).bernoulli_().to(device)
                    else:
                        c = torch.nn.functional.one_hot(torch.randint(0, NUM_CATEG, (BATCH_SIZE,)),
                                                        num_classes=NUM_CATEG).to(device)
                    z = torch.cat((c, _z), dim=1)
                else:
                    z = _z

                G_z = G(z)

                # G Loss
                # G wants to minimize value D(x) - D(G(z)) --> minimize the quantity -D(G(z))
                # (aka maximizing D(G(z)), i.e. tricking D to predict G(z) as positive/real)
                G_loss = torch.mean(-D(G_z))
                G_loss.backward(retain_graph=True)
                writer.add_scalar('Loss/Generator', G_loss.detach().item(), step)

                # Q Loss
                if train_Q:
                    Q_loss = criterion_Q(Q(G_z), c)
                    Q_loss.backward()
                    writer.add_scalar('Loss/Q_Network', Q_loss.detach().item(), step)
                    optimizer_Q.step()

                # Update
                optimizer_G.step()
            step += 1

        if not epoch % SAVE_INT:
            torch.save(G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_G.pt'))
            torch.save(D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_D.pt'))
            if train_Q:
                torch.save(Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Q.pt'))

            torch.save(optimizer_G.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Gopt.pt'))
            torch.save(optimizer_D.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Dopt.pt'))
            if train_Q:
                torch.save(optimizer_Q.state_dict(), os.path.join(logdir, f'epoch{epoch}_step{step}_Qopt.pt'))
