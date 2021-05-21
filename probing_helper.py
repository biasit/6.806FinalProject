import sys
import re
import matplotlib.pyplot as plt
import math
import time
from functools import reduce
import numpy as np
import os
import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from global_variables import *

# Get example from dataset
def get_example(dataset, idx):
  src_ids = dataset.src_ids[idx].unsqueeze(0).to(device)
  src_lens = torch.tensor(dataset.src_lens[idx]).unsqueeze(0).to(device)
  src_copy_ids = dataset.src_copy_ids[idx].unsqueeze(0).to(device)
  trg_ids = dataset.trg_ids[idx]
  trg_ids = trg_ids[trg_ids != PAD_INDEX].unsqueeze(0).to(device) # trim at first step for ease of analysis/speed
  src_text = dataset.src_sentences[idx]
  trg_text = dataset.trg_sentences[idx] #dataset.convert_from_copy_trg_vocab_to_orig(idx, trg_ids.squeeze(0).tolist())

  return src_ids, src_lens, src_copy_ids, trg_ids, src_text, trg_text

# Get activation for at each step (see e.g., Belnikov and Glass 2019)
def get_activation_mat(model, encoded_idx, encoded_idx_cp, lengths, trg_seq):
  """
    Get matrices of hidden states via teacher-forced decoding of a sentence.
    This allows for probing of the hidden state structure of the model.
    NOTE: this is copied from decoder.forward and can be incorporated directly into the model using, e.g., a "save_states" flag. However, I have chosen to go for clarity.
    - model: a copynet-(like) model with decoder as above
    - encoded_idx: [bs, seq_len] <- encoded indices
    - encoded_idx_cp: [bs, seq_len] <- encoded copy indices
    - trg_seq: [bs, max_len] <- target sequence indices (w/o copy indices for ease)
    - lengths: [bs] <- length of input sequence
  """

  # Setup
  seq_len = encoded_idx.size(1)
  max_len = trg_seq.size(1)
  bs = encoded_idx.size(0) # focus on 0 because this allows for less decoding
  V = model.decoder.vocab_size
  hs = model.decoder.hidden_size
  max_oovs = model.decoder.max_oovs
  trg_seq[trg_seq >= V] = UNK_INDEX # so no problems with decoder

  # Encoding
  encoded, _ = model.encode(encoded_idx, lengths)

  # Activation storage
  decoder_hs = torch.zeros((bs, max_len, hs), device = device)
  copy_weights = torch.zeros((bs, max_len, seq_len), device = device)

  # Initialize decoding
  context = None
  s_t = None
  outputs = torch.zeros((bs, max_len, V + max_oovs), device = device)

  # Pre-computing things in the encoder (for memory/computational efficiency)
  encoded_ohe = F.one_hot(encoded_idx_cp, num_classes = V + max_oovs).float()
  encoder_scores = model.decoder.tanh(model.decoder.Wo(encoded)) # [bs, seq_len, hidden_size]
  pad_mask = -1000 * (encoded_idx == PAD_INDEX)

  # Unroll the decoder RNN for `max_len` steps.
  for i in range(max_len):
    out_probs, s_t, context, prob_c = model.decoder.forward_step(trg_seq[:, i], encoded, encoded_idx, encoded_ohe, encoder_scores, pad_mask, prev_state = s_t, context = context)
    outputs[:, i, :] = out_probs

    # Storing stuff
    decoder_hs[:, i, :] = s_t
    copy_weights[:, i, :] = prob_c

  return outputs, encoded, decoder_hs, copy_weights

## Printing Hidden State Colors ----

# Print colored text
def colored(r, g, b, text):
  return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

# Print text colored according to value
def print_colored_text(txt_lst, val_lst, base_color = BASE_COLOR):
  col_txt = []
  for i, txt in enumerate(txt_lst):
    val = val_lst[i]
    rgb_val = math.floor(base_color * abs(val_lst[i]))
    if val > 0:
      col_txt.append(colored(0, 0, rgb_val, txt))
    else:
      col_txt.append(colored(rgb_val, 0, 0, txt))
  
  return col_txt # allow for flexibility of line breaks

def plot_hidden_state_activation(model, src_ids, src_copy_ids, src_lens, trg_ids, trg_text, neuron):
  """Plot activation of a particular neuron in a hidden state. All id inputs should be [1, *] as before. trg_text should be tokenized list."""
  
  _, _, decoder_hs = get_activation_mat(model, src_ids, src_copy_ids, src_lens, trg_ids)
  neuron_vals = decoder_hs[:, :, neuron].squeeze(0).detach().cpu().tolist()
  
  return print_colored_text(trg_text, neuron_vals)

# Add breaks to list of strings
def add_breaks(wd_list, break_every = 12):
  new_wd_lst = []
  for i, wd in enumerate(wd_list):
    if (i > 0) and (i % break_every == 0):
      new_wd_lst.append("\n")
    
    new_wd_lst.append(wd)

  return new_wd_lst

## Clustering Hidden States ----

# Randomly sample words from a cluster
def label_sample(dataset, tokens, labels, n_random = 5):
  k = len(np.unique(labels))

  for i in range(k):
    clust_tokens = tokens[labels == i]
    r_tokens = np.random.choice(clust_tokens, size = n_random, replace = False)

    r_wds = []
    for token in r_tokens:
      r_wds.append(dataset.trg_id2v[token])
    
    print(f"Random tokens from group {i} are: ", ", ".join(np.unique(r_wds)))
    print()

# Perform K-means clustering and random sampling
def kmeans_sample(dataset, hs_mat, tokens, k = 10, n_random = 5):
  kmeans = KMeans(n_clusters = k, random_state = SEED)
  kmeans_labels = kmeans.fit_predict(hs_mat)

  label_sample(dataset, tokens, kmeans_labels, n_random)

# Perform logistic regression on a symbol
def log_reg_symbol(dataset, hs_mat, trg_tokens, symbol, ntop = 10):
  # Get symbol
  symbol_idx = dataset.trg_v2id[symbol]
  is_symbol_vec = 1 * (trg_tokens == symbol_idx)

  # Logistic Regression
  log_reg = LogisticRegression(penalty = "l1", solver = "liblinear").fit(hs_mat, is_symbol_vec) # no convergence issues (likely better than L^2)
  top_neurons = np.argsort(np.abs(log_reg.coef_))[0, -ntop:]
  
  return log_reg, top_neurons