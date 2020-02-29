import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import os
from Constants import *
from DataSet.NameDS import NameDataset
from Model.NameLSTM import NameLSTM
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Convert import strings_to_index_tensor

EPOCH = 2000
PLOT_EVERY = 50
LR = 0.0005
CLIP = 1
NAME = "first"

def loss(self, Y_hat, Y):
    # TRICK 3 ********************************
    # before we calculate the negative log likelihood, we need to mask out the activations
    # this means we don't want to take into account padded items in the output vector
    # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
    # and calculate the loss on that.

    # flatten all the labels
    Y = Y.view(-1)

    # flatten all predictions
    Y_hat = Y_hat.view(-1, OUTPUT)

    # create a mask by filtering out all tokens that ARE NOT the padding token
    tag_pad_token = OUTPUT['<PAD>']
    mask = (Y > tag_pad_token).float()

    # count how many tokens we have
    nb_tokens = int(torch.sum(mask).data[0])

    # pick the values for the label and zero out the rest with the mask
    Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

    # compute cross entropy loss which ignores all <PAD> tokens
    ce_loss = -torch.sum(Y_hat) / nb_tokens

    return ce_loss

def plot_losses(loss, folder: str = "Results", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'b--', label="Cross Entropy Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

def save_weights(model: nn.Module, folder="Weights", filename=NAME):
    filepath = os.path.join(folder, f'{filename}.pth.tar')
    if not os.path.exists(folder):
        os.mkdir(folder)
    save_content = model.state_dict()
    torch.save(save_content, filepath)

def run_epochs(model: NameLSTM, iterator: DataLoader, optimizer: torch.optim.Optimizer,
               criterion: torch.nn.CrossEntropyLoss, clip: int):
    total_loss = 0
    all_losses = []
    for i in range(EPOCH):
        loss = train(model, iterator, optimizer, criterion, clip)
        total_loss += loss

        if i % PLOT_EVERY == 0:
            all_losses.append(total_loss / PLOT_EVERY)
            plot_losses(all_losses, filename="test")
            save_weights(model)

def train(model: NameLSTM, iterator: DataLoader, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss,
          clip: int):
    model.train()
    epoch_loss = 0

    for x in iterator:
        optimizer.zero_grad()

        max_len = len(max(x, key=len))
        src, src_len = strings_to_index_tensor(x, max_len, INPUT, IN_PAD_IDX)
        trg, _ = strings_to_index_tensor(x, max_len, OUTPUT, OUT_PAD_IDX)

        output = model.forward(src, src_len)

        loss = criterion(torch.transpose(output, 1, 2), trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss.item()

df = pd.read_csv('Data/FirstNames.csv')
ds = NameDataset(df, "name")
dl = DataLoader(ds, batch_size=256, shuffle=True)
model = NameLSTM(inputs=INPUT, outputs=OUTPUT)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.NLLLoss(ignore_index=OUT_PAD_IDX)

run_epochs(model, dl, optimizer, criterion, CLIP)