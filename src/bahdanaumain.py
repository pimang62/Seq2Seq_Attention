import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    TensorDataset,
    RandomSampler,
    DataLoader
    )
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

import random
import time

import os
import sys
sys.path.append(os.pardir)

from utils.preprocess import prepareData
from utils.dataloader import indexesFromSentence
from bahdanautrain import train
# from models.encoders import EncoderRNN
from models.bahdanauencoders import EncoderRNN
# from models.attention import AttnDecoderRNN
from models.bahdanauattention import AttnDecoderRNN
from eval import evaluate, evaluateRandomly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

# ---------------------------------------------------------------------

input_lang, output_lang, pairs = None, None, None
hidden_size, encoder1, attn_decoder1 = None, None, None

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepareData('eng', 'kor', True)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def main():
    hidden_size = 128
    batch_size = 32
    
    global input_lang, output_lang, pairs
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words).to(device)

    train(train_dataloader, encoder, decoder, 80, print_every=5, plot_every=5)

    evaluateRandomly(encoder1, attn_decoder1)

    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "Something wrong!")

    print(output_words)
    plt.matshow(attentions.numpy())
    
if __name__ == '__main__':
    main()


