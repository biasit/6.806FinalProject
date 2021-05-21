import sys
import pickle
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
from global_variables import *

# Update y based on scores at given time step
def update_y(current_score, out_probs, best_trellis, init = False):
  """
  For single sentence, update score and best y
    - current_score: [beam_size]
    - out_probs: [beam_size * (V + |X|)] <- probabilities on log scale (for computational stability due to lots of multiplication)
    - best_trellis: [beam_size, n_steps]  <- keeps track of best set of indices: expands by 1 on each turn
  """

  # Setup
  k = current_score.size(0)
  expanded_vocab_size = out_probs.size(0) // k # V'
  n_steps = best_trellis.size(1)

  # Mask out repeated upon initialization - don't want everything to be repeated (reduces to greedy search)
  if init:
    out_probs[expanded_vocab_size:] = -1000 # mask out repeated

  # Compute full scores up to this point
  expanded_score = torch.repeat_interleave(current_score, expanded_vocab_size) # [k * V']
  updated_score = out_probs + expanded_score

  # Find best options
  topk, topk_idx = tuple(torch.topk(updated_score, k)) # [k], [k]
  true_idx = topk_idx % expanded_vocab_size # [k]: true indices of the argmax
  which_node = topk_idx // expanded_vocab_size #torch.floor(topk_idx / expanded_vocab_size) # [k]: which original current_score they correspond to

  # Compute updated top-k scores and trellis
  final_score = torch.zeros(k, device = device)
  final_trellis = torch.zeros(k, n_steps + 1, device = device)

  for i in range(k):
    node = which_node[i]
    idx = true_idx[i].unsqueeze(0)

    final_trellis[i, :] = torch.cat((best_trellis[node, :], idx))
    final_score[i] = updated_score[topk_idx[i]]

  return final_score, final_trellis.type_as(best_trellis)

# Beam search (https://blog.ceshine.net/post/implementing-beam-search-part-1/#how-to-do-beam-search-efficiently)
# Note: I've assumed that bs = 1 here. The modification for bigger batches is straightforward, but the EOS exit condition means there likely won't be a gain in speed.
@torch.no_grad()
def beam_decode(model, src_ids, src_ids_cp, src_lengths, max_len = MAX_TRG_LENGTH, beam_size = 5):
  """Beam search decode a *single* sentence for CopyNet. Make sure to chop off the EOS token!"""

  # Setup and pre-computation
  bs, seq_len = src_ids.size() # bs = 1 everywhere

  if bs != 1:
    print("Warning: currently assumes batch_size = 1")

  V = model.decoder.vocab_size
  max_oovs = model.decoder.max_oovs

  # Encoding and getting first pass
  encoded, _ = model.encode(src_ids, src_lengths) # [bs, sl, 2 * h]

  # Initialize static components (in order of appearance in decoder); expand everything k times along batch dimension (for parallelization)
  encoded = torch.repeat_interleave(encoded, beam_size, 0) # [bs * k, sl, 2 * h]
  src_ids_cp = torch.repeat_interleave(src_ids_cp, beam_size, 0) # [bs * k, sl]
  src_ids_cp_ohe = F.one_hot(src_ids_cp, num_classes = V + max_oovs).float() # [bs * k, seq_len, V + |X|]
  encoder_scores = model.decoder.tanh(model.decoder.Wo(encoded)) # [bs * k, seq_len, h]
  pad_mask = -1000 * (src_ids_cp == PAD_INDEX) # [bs * k, sl]

  # Initialize dynamic components
  prev_y = torch.ones(bs * beam_size).fill_(START_INDEX).type_as(src_ids) # [bs * k]
  prev_state = None # default for first step of decoder
  context = None # default for first step of decoder

  # Storage
  best_trellis = prev_y.unsqueeze(1) # [bs * k, 1] - will vary in size over loop
  current_scores = torch.zeros(beam_size * bs, device = device)
  final_pred = torch.zeros(bs, seq_len, device = device)

  # Decoding loop (this is where bs = 1 assumption kicks in; else loop through bs and apply update_y)
  for i in range(max_len - 1):
    out_probs, prev_state, context, _ = model.decoder.forward_step(prev_y, encoded, src_ids_cp, src_ids_cp_ohe, encoder_scores,
                                                                pad_mask, prev_state, context)
    out_probs = out_probs.reshape(-1) # [k * V']

    # Updating scores and trellis
    current_scores, best_trellis = update_y(current_scores, torch.log(out_probs), best_trellis, i == 0) # [bs * k], [k, i + 1]
    prev_y = best_trellis[:, i + 1].contiguous().type_as(src_ids) # [k], continguous detaches slices from rest!

    prev_y[prev_y >= V] = UNK_INDEX # if copied, set to unknown for embedding!

    # Check exit condition (don't just check if *any* end, as this may be very bad; e.g., second step!)
    highest_score = torch.argmax(current_scores)

    if prev_y[highest_score] == END_INDEX:
      return best_trellis[highest_score] # don't cut off EOS token (the targets have them!)

    current_scores[prev_y == END_INDEX] = -1000 # cut off branches which have an EOS but are not chosen

  highest_score = torch.argmax(current_scores)
  return best_trellis[highest_score, :]
