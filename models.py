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


## Model Classes

# Bidirectional GRU encoder
class Encoder(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, max_src_length = MAX_SRC_LENGTH):
    """
    Inputs:
      - `vocab_size`: an int representing vocabulary size.
      - `embed_size`: an int representing embedding side (in *each* direction!).
      - `hidden_size`: an int representing the RNN hidden size.
    """
    super(Encoder, self).__init__()

    self.src_embed = nn.Embedding(vocab_size, embed_size)
    self.rnn = nn.GRU(embed_size, hidden_size, num_layers = 1, batch_first = True, bidirectional = True)
    self.max_src_length = max_src_length

  def forward(self, inputs, lengths):
    """
    Inputs:
      - `inputs`: a 2d-tensor of shape (batch_size, max_seq_length)
          representing a batch of padded embedded word vectors of source
          sentences.
      - `lengths`: a 1d-tensor of shape (batch_size,) representing the sequence
          lengths of `inputs`.

    Returns:
      - `outputs`: a 3d-tensor of shape
        (batch_size, max_seq_length, hidden_size). For now, this is a packed sequence.
      - `finals`: a 3d-tensor of shape (num_layers, batch_size, hidden_size).

    """
    lengths = lengths.cpu()
    inputs = self.src_embed(inputs) # [bs, seq_len, embed_size]
    packed_input = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths,
                                                           batch_first=True,
                                                           enforce_sorted=False)
    packed_outputs, h_n = self.rnn(packed_input) # second is [2, bs, hidden_size], first is packed
    outputs = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first = True, total_length = self.max_src_length)[0] # [bs, seq_len, 2 * hidden_size]

    return outputs, h_n

# Standard Seq2Seq Decoder with Bahdanau Attention
class Decoder(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size):
    """
      Inputs:
        - `vocab_size`: an int representing baseline vocab size
        - `max_oovs`: an int representing the maximum number of OOVS to be seen at testing time
    """
    super(Decoder, self).__init__()

    # Setup
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size

    # Ordinary layers
    self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=False) # bridge from bidirectional GRU to unidirectional (maybe won't be 2 * hidden_size for comp. purposes)
    self.trg_embed = nn.Embedding(vocab_size, embed_size)
    self.gru = nn.GRU(embed_size + 2 * hidden_size, hidden_size, batch_first=True) # 2 * hidden_size comes from attention - input to GRU is [embed; context]

    # Copy and attention weights
    self.Wg = nn.Linear(hidden_size, vocab_size) # generate weight (to apply to GRU state)
    self.Wa = nn.Linear(2 * hidden_size, hidden_size) # Bahdanau general attention weights

    # Non-linearities and helper functions
    self.tanh = nn.Tanh()

  #def generate_scores(self, encoded, encoded_idx, s_t):
  def generate_scores(self, s_t):
    """Get log probs for copy and generate mode. Mask the copy scores to avoid attending to padded stuff."""
    # encoder_scores: [bs, seq_len, hidden_size]  <- precomputed encoder scores (with Wo)
    # pad_mask: [bs, seq_len]  <- precomputed pad mask for encoder
    # s_t: [bs, h]  <- squeezed output of GRU at time t

    # First pass calculation
    psi_g = self.Wg(s_t) # [bs, V]
    prob_g = F.softmax(psi_g, dim = 1) # don't do log_softmax because matrix multiplication is messed up - fix this if underflow

    return prob_g

  def update_global_attention(self, encoded, encoder_scores, pad_mask, s_t):
    """ Take probabilities of all input indices"""
    psi_a = torch.bmm(encoder_scores, s_t.unsqueeze(2)).squeeze(2) # [bs, seq_len] WAS ordinary squeeze
    psi_a = psi_a + pad_mask # mask padded (basically zero out logits for these terms)

    prob_a = F.softmax(psi_a, dim = 1)
    prob_a = prob_a.unsqueeze(dim = 1) # [bs, 1, sl]

    return torch.bmm(prob_a, encoded) # [bs, 1, hiden_size*2]

  def forward_step(self, input_idx, encoded, pad_mask,
                   encoder_attention_scores=None, prev_state = None, context = None):
    """
      - input idx: [bs]  <- index of decoder input
      - encoded: [bs, seq_len, 2 * h]  <- output of encoder
      - encoded_idx: [bs, seq_len]  <- indices passed into encoder
      - encoded_ohe: [bs, seq_len, V + |X|]  <- pre-computed OHE source indices
      - encoder_scores: [bs, seq_len, hidden_size]  <- pre-computed encoder scores to dot product with decoder hidden state
      - pad_mask: [bs, seq_len]  <- pre-computed pad mask
      - prev_state: [1, bs, h]  <- previous hidden state
      - context: [bs, 1, 2 * h]  <- context vector computed using attention-like idea
    """

    # Initialization
    bs = encoded.size(0)
    seq_len = encoded.size(1)
    vocab_size = self.vocab_size
    hidden_size = self.hidden_size

    # Initialize state and attention if start token (MAY NEED TO MOVE STUFF TO GPU)
    if prev_state is None or context is None:
      final_encoded = encoded[:, -1, :] # [bs, 2 * h]
      prev_state = self.bridge(final_encoded).to(DEVICE) # [bs, h]
      context = torch.zeros((bs, 1, 2 * hidden_size), device = DEVICE) # [bs, 1, 2 * h]

    prev_state = prev_state.unsqueeze(0) # [1, bs, h], 1 due to num_layers * num_direction parameter

    # Update GRU state
    input_embed = self.trg_embed(input_idx).unsqueeze(1).to(DEVICE) # [bs, 1, embed_size]
    gru_input = torch.cat([input_embed, context], dim = 2) # [bs, 1, embed_size + 2 * h]
    _, s_t = self.gru(gru_input, prev_state) # [1, bs, h]
    s_t = s_t.squeeze(0) # [bs, h] WAS s_t.squeeze() - changed so that bs of 1 isn't messed up

    # Calculate scores, get output probability, and attention
    out_probs = self.generate_scores(s_t) # [bs, V + |X|], [bs, seq_len]
    context = self.update_global_attention(encoded, encoder_attention_scores, pad_mask, s_t)

    return out_probs, s_t, context

  def forward(self, inputs, encoded, encoded_idx, max_len = None, teacher_force=True):
    """
      Unroll the decoder one step at a time (with teacher forcing).
      - inputs: [bs, max_len] <- true sequence to decode
      - encoded: [bs, seq_len, 2 * h] <- encoded hidden states
      - encoded_idx: [bs, seq_len] <- encoded indices
      - max_len (optional): int  <- maximum length to unwind decoder

      Can be used for training or validation with or without teacher-forcing.
      Returns probability distributions over the V+max(|X|) vocabulary
    """

    if max_len is None:
      max_len = inputs.size(1)

    # Initialize
    bs = encoded.size(0)
    V = self.vocab_size
    context = None
    s_t = None
    outputs = torch.zeros((bs, max_len, V), device = DEVICE)

    # Pre-computing things in the encoder (for memory/computational efficiency)
    encoder_attention_scores = self.tanh(self.Wa(encoded)) # [bs, seq_len, hidden_size]
    pad_mask = -1000 * (encoded_idx == PAD_INDEX)

    # Unroll the decoder RNN for `max_len` steps.
    next_input = None
    for i in range(max_len):
      if i == 0 or teacher_force:
        next_input = inputs[:,i]
      else:
        next_input = torch.argmax(outputs[:,i-1,:], dim=-1)
        # Unk any out of vocabulary words
        next_input[next_input >= V] = UNK_INDEX

      out_probs, s_t, context = self.forward_step(next_input, encoded, pad_mask,
                                                  encoder_attention_scores=encoder_attention_scores,
                                                  prev_state = s_t, context = context)
      outputs[:, i, :] = out_probs

    return outputs

# CopyNet Decoder
# This implementation is heavily modeled off of: https://github.com/mjc92/CopyNet/blob/9e7f277e34208f871d449e3292ef24ad502b0d34/170704/copynet_dbg.py#L8
class CopyNetDecoder(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, max_oovs = MAX_SRC_LENGTH):
    """
      Inputs:
        - `vocab_size`: an int representing baseline vocab size
        - `max_oovs`: an int representing the maximum number of OOVS to be seen at testing time
    """
    super(CopyNetDecoder, self).__init__()

    # Setup
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.max_oovs = max_oovs

    # Ordinary layers
    self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=False) # bridge from bidirectional GRU to unidirectional (maybe won't be 2 * hidden_size for comp. purposes)
    self.trg_embed = nn.Embedding(vocab_size, embed_size)
    self.gru = nn.GRU(embed_size + 2 * hidden_size, hidden_size, batch_first=True) # 2 * hidden_size comes from attention - input to GRU is [embed; context]

    # Copy and attention weights
    self.Wg = nn.Linear(hidden_size, vocab_size) # generate weight (to apply to GRU state)
    self.Wo = nn.Linear(2 * hidden_size, hidden_size) # copy weight (to apply to encoder hidden state via quadratic form)

    # Non-linearities and helper functions
    self.tanh = nn.Tanh()

  #def generate_scores(self, encoded, encoded_idx, s_t):
  def generate_scores(self, encoder_scores, pad_mask, s_t):
    """Get log probs for copy and generate mode. Mask the copy scores to avoid attending to padded stuff."""
    # encoder_scores: [bs, seq_len, hidden_size]  <- precomputed encoder scores (with Wo)
    # pad_mask: [bs, seq_len]  <- precomputed pad mask for encoder
    # s_t: [bs, h]  <- squeezed output of GRU at time t

    # First pass calculation
    psi_g = self.Wg(s_t) # [bs, V]
    psi_c = torch.bmm(encoder_scores, s_t.unsqueeze(2)).squeeze(2) # [bs, seq_len] WAS ordinary squeeze
    psi_c = psi_c + pad_mask # mask padded (basically zero out logits for these terms)

    # Convert to probabilities
    prob_concat = torch.cat([psi_g, psi_c], dim = 1) # [bs, V + seq_len]: concatenate first so softmax shared between copy and generate
    prob_concat = F.softmax(prob_concat, dim = 1) # don't do log_softmax because matrix multiplication is messed up - fix this if underflow
    prob_g = prob_concat[:, :self.vocab_size]
    prob_c = prob_concat[:, self.vocab_size:]

    # Expand prob_g with some dummy terms
    oovs = torch.zeros(encoder_scores.size(0), self.max_oovs, device = DEVICE) + 1e-4 # 0 probabilities cause issues for NLLloss! Subtract off a large number if on log scale
    prob_g = torch.cat([prob_g, oovs], 1) # [bs, V + |X|]

    return prob_g, prob_c

  def combine_copy_gen(self, prob_g, prob_c, encoded_ohe): # used to pass in encoded_idx
    """
      Use encoded_idx (TODO: make compatible with target vocabulary) and copy/generated scores to get [bs, |V| + |X|] size vector of scores.
      |X| is the size of the unique additional vocabulary for that batch.
      Since we need to add the probabilities, we'll need to apply a log_softmax - we can then use np.logsumexp for numerically stable addition.
      The implementation above uses a bunch of OHE pushing - should be similar.
      Proposal: have a separate part of the batch with "unique" identifiers.
    """

    # OHE encoding and scattering scores
    prob_c_to_g = torch.bmm(prob_c.unsqueeze(dim = 1), encoded_ohe) # [bs, 1, V + |X|]
    prob_c_to_g = prob_c_to_g.squeeze(1) # [bs, V + |X|]

    return prob_g + prob_c_to_g # if on log scale, use logsumexp

  def update_attention(self, input_idx, encoded_idx, encoded, prob_c):
    """ Use ``selective reading`` to get weights for next step (take normalized copy probabilities of relevant indices)"""

    bs, seq_len = encoded_idx.size()

    # Loop through to check which indices are relevant: take copy weights from those
    relevant_idx = torch.zeros((bs, seq_len), device = DEVICE)
    for i in range(bs):
      relevant_idx[i, :] = 1 * (encoded_idx[i, :] == input_idx[i])

    # Get attention via copy weights
    copy_weight = relevant_idx * prob_c # [bs, sl]
    copy_weight = F.normalize(copy_weight, p = 1, dim = 1) # row-normalize copy weights
    copy_weight = copy_weight.unsqueeze(dim = 1) # [bs, 1, sl]

    return torch.bmm(copy_weight, encoded) # [bs, 1, 2 * h]

  def forward_step(self, input_idx, encoded, encoded_idx, encoded_ohe, encoder_scores, pad_mask, prev_state = None, context = None):
    """
      - input idx: [bs]  <- index of decoder input
      - encoded: [bs, seq_len, 2 * h]  <- output of encoder
      - encoded_idx: [bs, seq_len]  <- indices passed into encoder
      - encoded_ohe: [bs, seq_len, V + |X|]  <- pre-computed OHE source indices
      - encoder_scores: [bs, seq_len, hidden_size]  <- pre-computed encoder scores to dot product with decoder hidden state
      - pad_mask: [bs, seq_len]  <- pre-computed pad mask
      - prev_state: [1, bs, h]  <- previous hidden state
      - context: [bs, 1, 2 * h]  <- context vector computed using attention-like idea
    """

    # Initialization
    bs = encoded.size(0)
    seq_len = encoded.size(1)
    vocab_size = self.vocab_size
    hidden_size = self.hidden_size

    # Initialize state and attention if start token (MAY NEED TO MOVE STUFF TO GPU)
    if prev_state is None or context is None:
      final_encoded = encoded[:, -1, :] # [bs, 2 * h]
      prev_state = self.bridge(final_encoded).to(DEVICE) # [bs, h]
      context = torch.zeros((bs, 1, 2 * hidden_size), device = DEVICE) # [bs, 1, 2 * h]

    prev_state = prev_state.unsqueeze(0) # [1, bs, h], 1 due to num_layers * num_direction parameter

    # Update GRU state
    input_embed = self.trg_embed(input_idx).unsqueeze(1).to(DEVICE) # [bs, 1, embed_size]
    gru_input = torch.cat([input_embed, context], dim = 2) # [bs, 1, embed_size + 2 * h]
    _, s_t = self.gru(gru_input, prev_state) # [1, bs, h]
    s_t = s_t.squeeze(0) # [bs, h] WAS s_t.squeeze() - changed so that bs of 1 isn't messed up

    # Calculate scores, get output probability, and attention
    prob_g, prob_c = self.generate_scores(encoder_scores, pad_mask, s_t) # [bs, V + |X|], [bs, seq_len]
    out_probs = self.combine_copy_gen(prob_g, prob_c, encoded_ohe) # [bs, V + |X|]
    context = self.update_attention(input_idx, encoded_idx, encoded, prob_c) # [bs, 1, 2 * h]

    return out_probs, s_t, context, prob_c # NOTE: ADDED PROB_C

  def forward(self, inputs, encoded, encoded_idx, max_len = None):
    """
      Unroll the decoder one step at a time (with teacher forcing).
      - inputs: [bs, max_len] <- true sequence to decode
      - encoded: [bs, seq_len, 2 * h] <- encoded hidden states
      - encoded_idx: [bs, seq_len] <- encoded indices
      - max_len (optional): int  <- maximum length to unwind decoder
    """

    if max_len is None:
      max_len = inputs.size(1)

    # Initialize
    bs = encoded.size(0)
    V = self.vocab_size
    context = None
    s_t = None
    outputs = torch.zeros((bs, max_len, V + self.max_oovs), device = DEVICE)

    # Pre-computing things in the encoder (for memory/computational efficiency)
    encoded_ohe = F.one_hot(encoded_idx, num_classes = V + self.max_oovs).float()
    encoder_scores = self.tanh(self.Wo(encoded)) # [bs, seq_len, hidden_size]
    pad_mask = -1000 * (encoded_idx == PAD_INDEX)

    # Unroll the decoder RNN for `max_len` steps.
    for i in range(max_len):
      out_probs, s_t, context, _ = self.forward_step(inputs[:, i], encoded, encoded_idx, encoded_ohe, encoder_scores, pad_mask, prev_state = s_t, context = context)
      outputs[:, i, :] = out_probs

    return outputs

# Decoder with copy mechanism and attention
class CopyNetDecoderWithAttention(nn.Module):
  def __init__(self, vocab_size, embed_size, hidden_size, max_oovs = MAX_SRC_LENGTH):
    """
      Inputs:
        - `vocab_size`: an int representing baseline vocab size
        - `max_oovs`: an int representing the maximum number of OOVS to be seen at testing time
    """
    super(CopyNetDecoderWithAttention, self).__init__()

    # Setup
    self.vocab_size = vocab_size
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.max_oovs = max_oovs

    # Ordinary layers
    self.bridge = nn.Linear(2 * hidden_size, hidden_size, bias=False) # bridge from bidirectional GRU to unidirectional (maybe won't be 2 * hidden_size for comp. purposes)
    self.trg_embed = nn.Embedding(vocab_size, embed_size)
    self.gru = nn.GRU(embed_size + 2 * hidden_size, hidden_size, batch_first=True) # 2 * hidden_size comes from attention - input to GRU is [embed; context]

    # Copy and attention weights
    self.Wg = nn.Linear(hidden_size, vocab_size) # generate weight (to apply to GRU state)
    self.Wo = nn.Linear(2 * hidden_size, hidden_size) # copy weight (to apply to encoder hidden state via quadratic form)
    self.Wa = nn.Linear(2 * hidden_size, hidden_size) # Bahdanau general attention weights

    # Non-linearities and helper functions
    self.tanh = nn.Tanh()

  #def generate_scores(self, encoded, encoded_idx, s_t):
  def generate_scores(self, encoder_scores, pad_mask, s_t):
    """Get log probs for copy and generate mode. Mask the copy scores to avoid attending to padded stuff."""
    # encoder_scores: [bs, seq_len, hidden_size]  <- precomputed encoder scores (with Wo)
    # pad_mask: [bs, seq_len]  <- precomputed pad mask for encoder
    # s_t: [bs, h]  <- squeezed output of GRU at time t

    # First pass calculation
    psi_g = self.Wg(s_t) # [bs, V]
    #print(s_t.size())
    psi_c = torch.bmm(encoder_scores, s_t.unsqueeze(2)).squeeze(2) # [bs, seq_len] WAS ordinary squeeze
    psi_c = psi_c + pad_mask # mask padded (basically zero out logits for these terms)

    # Convert to probabilities
    prob_concat = torch.cat([psi_g, psi_c], dim = 1) # [bs, V + seq_len]: concatenate first so softmax shared between copy and generate
    prob_concat = F.softmax(prob_concat, dim = 1) # don't do log_softmax because matrix multiplication is messed up - fix this if underflow
    prob_g = prob_concat[:, :self.vocab_size]
    prob_c = prob_concat[:, self.vocab_size:]

    # Expand prob_g with some dummy terms
    oovs = torch.zeros(encoder_scores.size(0), self.max_oovs, device = DEVICE) + 1e-4 # 0 probabilities cause issues for NLLloss! Subtract off a large number if on log scale
    prob_g = torch.cat([prob_g, oovs], 1) # [bs, V + |X|]

    return prob_g, prob_c

  def combine_copy_gen(self, prob_g, prob_c, encoded_ohe): # used to pass in encoded_idx
    """
      Use encoded_idx (TODO: make compatible with target vocabulary) and copy/generated scores to get [bs, |V| + |X|] size vector of scores.
      |X| is the size of the unique additional vocabulary for that batch.
      Since we need to add the probabilities, we'll need to apply a log_softmax - we can then use np.logsumexp for numerically stable addition.
      The implementation above uses a bunch of OHE pushing - should be similar.
      Proposal: have a separate part of the batch with "unique" identifiers.
    """

    # OHE encoding and scattering scores
    prob_c_to_g = torch.bmm(prob_c.unsqueeze(dim = 1), encoded_ohe) # [bs, 1, V + |X|]
    prob_c_to_g = prob_c_to_g.squeeze(1) # [bs, V + |X|]

    return prob_g + prob_c_to_g # if on log scale, use logsumexp

  def update_global_attention(self, encoded, encoder_scores, pad_mask, s_t):
    """ Take probabilities of all input indices"""
    psi_a = torch.bmm(encoder_scores, s_t.unsqueeze(2)).squeeze(2) # [bs, seq_len] WAS ordinary squeeze
    psi_a = psi_a + pad_mask # mask padded (basically zero out logits for these terms)

    prob_a = F.softmax(psi_a, dim = 1)
    prob_a = prob_a.unsqueeze(dim = 1) # [bs, 1, sl]

    return torch.bmm(prob_a, encoded) # [bs, 1, hiden_size*2]

  def forward_step(self, input_idx, encoded, encoded_idx, encoded_ohe, encoder_scores, pad_mask,
                   encoder_attention_scores=None, prev_state = None, context = None):
    """
      - input idx: [bs]  <- index of decoder input
      - encoded: [bs, seq_len, 2 * h]  <- output of encoder
      - encoded_idx: [bs, seq_len]  <- indices passed into encoder
      - encoded_ohe: [bs, seq_len, V + |X|]  <- pre-computed OHE source indices
      - encoder_scores: [bs, seq_len, hidden_size]  <- pre-computed encoder scores to dot product with decoder hidden state
      - pad_mask: [bs, seq_len]  <- pre-computed pad mask
      - prev_state: [1, bs, h]  <- previous hidden state
      - context: [bs, 1, 2 * h]  <- context vector computed using attention-like idea
    """

    # Initialization
    bs = encoded.size(0)
    seq_len = encoded.size(1)
    vocab_size = self.vocab_size
    hidden_size = self.hidden_size

    # Initialize state and attention if start token (MAY NEED TO MOVE STUFF TO GPU)
    if prev_state is None or context is None:
      final_encoded = encoded[:, -1, :] # [bs, 2 * h]
      prev_state = self.bridge(final_encoded).to(DEVICE) # [bs, h]
      context = torch.zeros((bs, 1, 2 * hidden_size), device = DEVICE) # [bs, 1, 2 * h]

    prev_state = prev_state.unsqueeze(0) # [1, bs, h], 1 due to num_layers * num_direction parameter

    # Update GRU state
    input_embed = self.trg_embed(input_idx).unsqueeze(1).to(DEVICE) # [bs, 1, embed_size]
    gru_input = torch.cat([input_embed, context], dim = 2) # [bs, 1, embed_size + 2 * h]
    _, s_t = self.gru(gru_input, prev_state) # [1, bs, h]
    s_t = s_t.squeeze(0) # [bs, h] WAS s_t.squeeze() - changed so that bs of 1 isn't messed up

    # Calculate scores, get output probability, and attention
    prob_g, prob_c = self.generate_scores(encoder_scores, pad_mask, s_t) # [bs, V + |X|], [bs, seq_len]
    #print(f"Allocated {torch.cuda.memory_allocated(0)/(1024**3)} GB after generating scores")
    out_probs = self.combine_copy_gen(prob_g, prob_c, encoded_ohe) # [bs, V + |X|]
    #print(f"Allocated {torch.cuda.memory_allocated(0)/(1024**3)} GB after computing output probabilities")
    # context = self.update_attention(input_idx, encoded_idx, encoded, prob_c) # [bs, 1, 2 * h]
    context = self.update_global_attention(encoded, encoder_attention_scores, pad_mask, s_t)
    #print(f"Allocated {torch.cuda.memory_allocated(0)/(1024**3)} GB after updating attention")

    return out_probs, s_t, context

  def forward(self, inputs, encoded, encoded_idx, max_len = None, teacher_force=True):
    """
      Unroll the decoder one step at a time (with teacher forcing).
      - inputs: [bs, max_len] <- true sequence to decode
      - encoded: [bs, seq_len, 2 * h] <- encoded hidden states
      - encoded_idx: [bs, seq_len] <- encoded indices
      - max_len (optional): int  <- maximum length to unwind decoder

      Can be used for training or validation with or without teacher-forcing.
      Returns probability distributions over the V+max(|X|) vocabulary
    """

    if max_len is None:
      max_len = inputs.size(1)

    # Initialize
    bs = encoded.size(0)
    V = self.vocab_size
    context = None
    s_t = None
    outputs = torch.zeros((bs, max_len, V + self.max_oovs), device = DEVICE)

    # Pre-computing things in the encoder (for memory/computational efficiency)
    encoded_ohe = F.one_hot(encoded_idx, num_classes = V + self.max_oovs).float()
    encoder_scores = self.tanh(self.Wo(encoded)) # [bs, seq_len, hidden_size]
    encoder_attention_scores = self.tanh(self.Wa(encoded)) # [bs, seq_len, hidden_size]
    pad_mask = -1000 * (encoded_idx == PAD_INDEX)

    # Unroll the decoder RNN for `max_len` steps.
    next_input = None
    for i in range(max_len):
      if i == 0 or teacher_force:
        next_input = inputs[:,i]
      else:
        next_input = torch.argmax(outputs[:,i-1,:], dim=-1)
        # Unk any out of vocabulary words
        next_input[next_input >= V] = UNK_INDEX

      out_probs, s_t, context = self.forward_step(next_input, encoded, encoded_idx,
                                                  encoded_ohe, encoder_scores, pad_mask,
                                                  encoder_attention_scores=encoder_attention_scores,
                                                  prev_state = s_t, context = context)
      outputs[:, i, :] = out_probs

    return outputs

# Generic combination of Encoder-Decoder
class Seq2Seq(nn.Module):
  def __init__(self, encoder, decoder):
    super(Seq2Seq, self).__init__()

    self.encoder = encoder
    self.decoder = decoder

  def encode(self, encoded_idx, lengths):
    return self.encoder(encoded_idx, lengths)

  def decode(self, inputs, encoded, encoded_idx_cp, max_len=None):
    return self.decoder(inputs, encoded, encoded_idx_cp, max_len=None)

  def forward(self, encoded_idx, encoded_idx_cp, trg_seq, lengths, max_len=None):
    encoded, _ = self.encode(encoded_idx, lengths)
    return self.decode(trg_seq[:, :-1], encoded, encoded_idx_cp, max_len) # taking off last token from HW3


### Below is deprected code for a Transformer Encoder
class CustomTransformerEncoder(nn.Module):
    def __init__(self, embed_size=50, nheads=2, nlayers=6, vocab_size=500, max_input_length=700):
        super(CustomTransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.src_embed = nn.Embedding(vocab_size, embed_size)

        self.nheads = nheads
        self.nlayers = nlayers
        self.max_input_length = max_input_length

        self.pos_encoder = PositionalEncoding(self.embed_size, max_len=self.max_input_length)
        self.t_layer = nn.TransformerEncoderLayer(self.embed_size, self.nheads)
        self.transformer_encoder = nn.TransformerEncoder(self.t_layer, self.nlayers)

    def forward(self, src, lengths):
        # input should be:
        # src: (batch_size, max_seq_len)
        # lengths: (batch_size)  - should be unneeded
        #
        # same outputs as the HW2 encoder from above

        # get correct mask
        src_mask = src.float().masked_fill(src == PAD_INDEX, float('-inf')).masked_fill(src != PAD_INDEX, 0.).to(DEVICE)

        src = self.src_embed(src) # [batch_size x max_seq_len x embed_size]

        src = src.transpose(0,1) # convert input to batch second [max_seq_len x batch_size x encoding_dim]
        src_mask = src_mask.unsqueeze(0).transpose(0,1).repeat(self.nheads,src_mask.size(1),1) # convert mask to correct shape (repeated...) [nheads*batch_size x max_seq_len x max_seq_len]

        src *= math.sqrt(self.embed_size) # they recommend this to reduce variance in embedding
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask) # [max_seq_len x batch_size x encoding_dim]
        output = output.transpose(0,1)                   # [batch_size x max_seq_len x encoding_dim]
        # final_output = torch.max(output, dim=1)[0].squeeze() # [batch_size x encoding_dim]
        final_output = torch.zeros(len(lengths), self.embed_size).to(DEVICE)
        for i in range(len(lengths)):
            final_output[i] = output[i,lengths[i]-1]
        return output, final_output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=700):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model) # [max_len x d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len x 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model / 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1) # [max_len x 1 x d_model ]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # input should be (seq_len, batch_size, encoding_dim)
        return x + self.pe[:x.size(0), :]
