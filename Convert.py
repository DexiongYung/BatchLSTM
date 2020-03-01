import torch
from Constants import *

def strings_to_index_tensor(strings: list, max_string_len: int, vocab: dict, pad: str) -> list:
    """
    Turn a list of strings into a tensor of shape: <max_string_len x batch_size (length of strings)>.
    index_function should be a function that converts a character into an appropriate index.
    Example: strings: ["012","9 ."], max_string_len: 4,
            => torch.tensor([[0,9],[1,10],[2,11],[10,10]])
    """
    tensor = torch.ones(len(strings),max_string_len).type(torch.LongTensor) * pad
    lens = torch.zeros(len(strings)).type(torch.LongTensor)
    for i_s, s in enumerate(strings):
        lens[i_s] = len(s)
        for i_char, char in enumerate(s):
            tensor[i_s][i_char] = vocab[char]
    return tensor.to(DEVICE), lens.to(DEVICE)