import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt

import random
import time

import os
import sys
sys.path.append(os.pardir)


from utils.preprocess import prepareData
from train import trainIters
from models.encoders import EncoderRNN
from models.attention import AttnDecoderRNN
# from models.bahdanau import AttnDecoderRNN
from eval import evaluate, evaluateRandomly
from plot import evaluateAndShowAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------

input_lang, output_lang, pairs = None, None, None
hidden_size, encoder1, attn_decoder1 = None, None, None

def main():
    global input_lang, output_lang, pairs
    input_lang, output_lang, pairs = prepareData('eng', 'kor', False)
    
    global hidden_size, encoder1, attn_decoder1
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(input_lang, output_lang, pairs, encoder1, attn_decoder1, 75000, print_every=5000)

    # evaluateRandomly(input_lang, output_lang, pairs, encoder1, attn_decoder1)
    
    """infer"""
    evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, "I am nice.")

    # evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, "something wrong")

    # evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, "I think I'm sleeping")

    evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, "I go to sleep!")
    

if __name__ == '__main__':
    main()