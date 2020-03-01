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
import Utilities.JSON as config
import argparse
from Convert import strings_to_index_tensor

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='First', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--batch_size', help='Size of the batch training on', nargs='?', default=500, type=int)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=5000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--embed_dim', help='Word embedding size', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)


# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
EPOCH = args.num_epochs
PLOT_EVERY = 50
NUM_LAYERS = args.num_layers
LR = args.lr
HIDDEN_SZ = args.hidden_size
CLIP = 1
EMBED_DIM = args.embed_dim
TRAIN_FILE = args.train_file
COLUMN = args.column

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

        output, hidden = model.forward(src, src_len)

        loss = criterion(torch.transpose(output, 1, 2), trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return loss.item()

def train_rnn(model: NameLSTM, name: str, optimizer: torch.optim.Optimizer, criterion: torch.nn.CrossEntropyLoss):
    model.train()
    optimizer.zero_grad()
    hidden = model.init_hidden(1)
    src, _ = strings_to_index_tensor(name, max_len, INPUT, IN_PAD_IDX)
    trg, _ = strings_to_index_tensor(name, max_len, OUTPUT, OUT_PAD_IDX)

    for i in range(len(name)):
        max_len = len(name)

        output, hidden = model.forward(src, hidden)

        loss = criterion(output, trg[i])

    return loss.item()

def generate_name(model: NameLSTM):
    output, hidden = model.forward(torch.tensor([[0]]).to(DEVICE),[1])

    return output

to_save = {
    'session_name': NAME,
    'hidden_size': args.hidden_size,
    'batch_size': args.batch_size,
    'input_size': len(INPUT),
    'output_size': len(OUTPUT),
    'input': INPUT,
    'output': OUTPUT,
    'num_layers': NUM_LAYERS,
    'embed_size': EMBED_DIM
}

config.save_json(f'Config/{NAME}.json', to_save)


df = pd.read_csv(TRAIN_FILE)
ds = NameDataset(df, COLUMN)
dl = DataLoader(ds, batch_size=256, shuffle=True)
model = NameLSTM(INPUT, OUTPUT, HIDDEN_SZ, NUM_LAYERS, EMBED_DIM)
optimizer = optim.Adam(model.parameters(), lr=LR)
model.to(DEVICE)
criterion = nn.NLLLoss(ignore_index=OUT_PAD_IDX)

generate_name(model)