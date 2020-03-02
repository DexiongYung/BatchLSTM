import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import Utilities.JSON as config
from Constants import *
from Convert import strings_to_index_tensor
from DataSet.NameDS import NameDataset
from Model.LSTM import LSTM

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='first_lstm', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.05, type=float)
parser.add_argument('--batch_size', help='Size of the batch training on', nargs='?', default=500, type=int)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=5000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--embed_dim', help='Word embedding size', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print_every', help='Number of iterations till print', nargs='?', default=50, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
EPOCH = args.num_epochs
PLOT_EVERY = args.print_every
NUM_LAYERS = args.num_layers
LR = args.lr
HIDDEN_SZ = args.hidden_size
TRAIN_FILE = args.train_file
COLUMN = args.column
INPUT = INPUT
OUTPUT = INPUT
INPUT_SZ = len(INPUT)
OUTPUT_SZ = len(OUTPUT)
MAX_LEN = 20

to_save = {
    'session_name': NAME,
    'hidden_size': args.hidden_size,
    'batch_size': args.batch_size,
    'num_layers': NUM_LAYERS,
    'input_size': INPUT_SZ,
    'output_size': OUTPUT_SZ,
    'input': INPUT,
    'output': OUTPUT,
}

config.save_json(f'Config/{NAME}.json', to_save)

def init_input():
    decoder_input = torch.zeros(1, 1, OUTPUT_SZ)
    decoder_input[0, 0, OUTPUT['<SOS>']] = 1.
    return decoder_input.to(DEVICE)

def plot_losses(loss, folder: str = "Results"):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'b--', label="Cross Entropy Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{NAME}")
    plt.close()


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    line_list = ['<SOS>']

    for idx in range(len(line)):
        line_list.append(line[idx])

    tensor = torch.zeros(len(line_list), 1, INPUT_SZ)

    for idx in range(len(line_list)):
        letter = line_list[idx]
        tensor[idx][0][INPUT[letter]] = 1

    return tensor.to(DEVICE)

def inputCharTensor(line):
    line_list = []

    for idx in range(len(line)):
        line_list.append(line[idx])

    tensor = torch.zeros(len(line_list), 1, INPUT_SZ)

    for idx in range(len(line_list)):
        letter = line_list[idx]
        tensor[idx][0][INPUT[letter]] = 1

    return tensor.to(DEVICE)

def list_to_tensor(input: list) -> torch.Tensor:
    tensor = torch.zeros(len(input), 1, INPUT_SZ)
    for i in range(len(input)):
        tensor[i, 0, INPUT[input[i]]] = 1
    return tensor.to(DEVICE)

def targetTensor(line):
    letter_indexes = [OUTPUT[line[li]] for li in range(len(line))]
    letter_indexes.append(OUTPUT['<EOS>'])  # EOS

    return torch.LongTensor(letter_indexes).to(DEVICE)

def sample(lstm: LSTM):
    with torch.no_grad():  # no need to track history in sampling
        input = init_input()
        hidden = model.initHidden()
        hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
        name = ''
        char = '<SOS>'
        iter = 0

        while char is not '<EOS' and iter < MAX_LEN:
            iter += 1
            probs, hidden = model(input, hidden)
            best_index = torch.argmax(probs, dim=2).item()
            for k, v in INPUT.items():
                if best_index == v:
                    name += k
            input = torch.zeros(1, 1, INPUT_SZ).to(DEVICE)
            input[0, 0, best_index] = 1.
        
        return name

def train(x: str, model: LSTM):
    optimizer.zero_grad()

    loss = 0.

    list_x = []

    for i in range(len(x)):
        list_x.append(x[i])

    x = list_to_tensor(list_x)
    input = init_input()
    hidden = model.initHidden()
    hidden = (hidden[0].to(DEVICE), hidden[1].to(DEVICE))
    name = ''

    for i in range(x.shape[0]):
        probs, hidden = model(input, hidden)
        _, nonzero_indexes = x[i].topk(1)
        best_index = torch.argmax(probs, dim=2).item()
        loss += criterion(probs[0], nonzero_indexes[0])
        for k, v in INPUT.items():
            if best_index == v:
                name += k
        input = torch.zeros(1, 1, INPUT_SZ).to(DEVICE)
        input[0, 0, best_index] = 1.

    loss.backward()
    optimizer.step()
    return name, loss.item()

def run_epochs(dl: DataLoader, model: LSTM):
    total_loss = 0
    all_losses = []
    iter = 0
    for i in range(EPOCH):
        for x in dl:
            iter += 1
            output, loss = train(x[0], model)
            total_loss += loss

            if iter % PLOT_EVERY == 0:
                all_losses.append(total_loss / PLOT_EVERY)
                total_loss = 0
                plot_losses(all_losses)
                torch.save({'weights': model.state_dict()}, os.path.join(f"Weights/{NAME}.path.tar"))


df = pd.read_csv(TRAIN_FILE)
ds = NameDataset(df, COLUMN)
dl = DataLoader(ds, batch_size=1, shuffle=True)
model = LSTM(input_size=INPUT_SZ, num_layers=NUM_LAYERS, output_size=OUTPUT_SZ, hidden_sz=HIDDEN_SZ)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.NLLLoss()
run_epochs(dl, model)

