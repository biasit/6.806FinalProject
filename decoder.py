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
  current_scores = torch.zeros(beam_size * bs, device = DEVICE)
  final_pred = torch.zeros(bs, seq_len, device = DEVICE)

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

@torch.no_grad()
def beam_decode_with_attention(model, src_ids, src_ids_cp, src_lengths, max_len = MAX_TRG_LENGTH, beam_size = 5):
  """Beam search decode (assumes a model with attention) a *single* sentence for CopyNet.
     Make sure to chop off the EOS token!"""

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
  encoder_attention_scores = model.decoder.tanh(model.decoder.Wa(encoded)) # [bs, seq_len, hidden_size]


  pad_mask = -1000 * (src_ids_cp == PAD_INDEX) # [bs * k, sl]

  # Initialize dynamic components
  prev_y = torch.ones(bs * beam_size).fill_(START_INDEX).type_as(src_ids) # [bs * k]
  prev_state = None # default for first step of decoder
  context = None # default for first step of decoder

  # Storage
  best_trellis = prev_y.unsqueeze(1) # [bs * k, 1] - will vary in size over loop
  current_scores = torch.zeros(beam_size * bs, device = device)
  final_pred = torch.zeros(bs, seq_len, device = device)
  #print("FINISHED SETUP")

  # Decoding loop (this is where bs = 1 assumption kicks in; else loop through bs and apply update_y)
  for i in range(max_len - 1):
    #print(i)
    out_probs, prev_state, context = model.decoder.forward_step(prev_y, encoded, src_ids_cp, src_ids_cp_ohe, encoder_scores,
                                                                pad_mask,
                                                                encoder_attention_scores=encoder_attention_scores,
                                                                prev_state = prev_state, context = context)

    #print(prev_state.size())
    #print(context.size())
    out_probs = out_probs.reshape(-1) # [k * V']
    #print("FINISHED DECODING STEP")

    # Updating scores and trellis
    # print(out_probs.size())
    # print()
    # print(best_trellis.size())
    current_scores, best_trellis = update_y(current_scores, torch.log(out_probs), best_trellis, i == 0) # [bs * k], [k, i + 1]
    #print(current_scores.size(), best_trellis.size())
    prev_y = best_trellis[:, i + 1].contiguous().type_as(src_ids) # [k], continguous detaches slices from rest!

    ## Check copying
    # for y in prev_y:
    #   if y >= V:
    #     print(best_trellis)
    #     print(i)
    #     print(y)
    #print(best_trellis)
    ##

    prev_y[prev_y >= V] = UNK_INDEX # if copied, set to unknown for embedding!

    # Check exit condition (don't just check if *any* end, as this may be very bad; e.g., second step!)
    highest_score = torch.argmax(current_scores)

    if prev_y[highest_score] == END_INDEX:
      return best_trellis[highest_score] # don't cut off EOS token (the targets have them!)

    current_scores[prev_y == END_INDEX] = -1000 # cut off branches which have an EOS but are not chosen

  highest_score = torch.argmax(current_scores)
  return best_trellis[highest_score, :]


# Get the model's output - teacher forcing - get perplexity while at it
def validation_predictions(data_loader, model, loss_compute):
    model.eval()
    total_tokens = 0
    total_loss = 0
    predictions = []
    for i, (src_ids_BxT, src_lengths_B, src_ids_cp_BxT, trg_ids_BxL, trg_lengths_B) in enumerate(data_loader):
        src_ids_BxT = src_ids_BxT.to(device)
        src_lengths_B = src_lengths_B.to(device)
        src_ids_cp_BxT = src_ids_cp_BxT.to(device)
        trg_ids_BxL = trg_ids_BxL.to(device)

        # Unk any out of vocabulary words
        actual_trg_ids = torch.where(trg_ids_BxL >= model.decoder.vocab_size,
                                     torch.ones(trg_ids_BxL.shape).long().to(device),
                                     trg_ids_BxL)

        output = model(src_ids_BxT, src_ids_cp_BxT, actual_trg_ids, src_lengths_B) # copy indices shouldn't mess up trg_id evaluation because tacked on at end!

        loss = loss_compute(x=output, y=trg_ids_BxL[:, 1:], norm=src_ids_BxT.size(0))

        total_loss += loss
        total_tokens += (trg_ids_BxL[:, 1:] != PAD_INDEX).data.sum().item()

        # get predictions
        curr_predictions = torch.argmax(output,dim=-1).squeeze().detach().cpu()
        for i_c, cp in enumerate(curr_predictions):
            cp = cp.tolist()
            predictions.append([START_INDEX]+cp[:trg_lengths_B[i_c]-1])

    return math.exp(total_loss / float(total_tokens)), predictions

def write_to_file(file_name, trg_sentences):
    # Get train target target
    with open(file_name, 'w') as f:
        for i_s, sent in enumerate(trg_sentences):
            f.write(' '.join(sent))
            if i_s < len(trg_sentences) - 1:
              f.write('\n')

def store_model_predictions(model, val_set, save_name, attention=False):
    evaluation_dir = "./evaluation_files"

    # Get perplexity and predictions if desired (can print out)
    from train_helper import SimpleLossCompute
    val_data_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1,
                                      shuffle=False)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
    val_ppl, val_predictions = validation_predictions(val_data_loader, model, loss_compute=SimpleLossCompute(criterion, model, None))

    # Get tokenized predictions
    val_predictions_tokens = []
    for i_v, val_p in enumerate(val_predictions):
        curr_tok = val_set.convert_from_copy_trg_vocab_with_src(val_set.src_sentences[i_v], val_p)
        val_predictions_tokens.append(curr_tok)

    # Store predictions
    val_predictions_teacher_force_dir = os.path.join(evaluation_dir, save_name+'_trainforce.txt')
    write_to_file(val_predictions_teacher_force_dir, val_predictions_tokens)

    # Get beam predictions
    val_predictions_beam = []
    for i in range(len(val_set.trg_sentences)):
        ex_src_ids = val_set.src_ids[i].unsqueeze(0).to(DEVICE)
        ex_src_lens = torch.tensor(val_set.src_lens[i]).unsqueeze(0).to(DEVICE)
        ex_src_copy_ids = val_set.src_copy_ids[i].unsqueeze(0).to(DEVICE)

        # Beam Search
        if attention:
            decoded = beam_decode_with_attention(model, ex_src_ids, ex_src_copy_ids, ex_src_lens)
        else:
            decoded = beam_decode(model, ex_src_ids, ex_src_copy_ids, ex_src_lens)
        curr_p = val_set.convert_from_copy_trg_vocab_with_src(val_set.src_sentences[i],
                                                                decoded.detach().cpu().tolist())
        if i % 500 == 0:
            print("Prop done:", i / len(val_set.trg_sentences))
        val_predictions_beam.append(curr_p)

    val_predictions_beam_dir = os.path.join(evaluation_dir, save_name+'_beam.txt')
    write_to_file(val_predictions_beam_dir, val_predictions_beam)
