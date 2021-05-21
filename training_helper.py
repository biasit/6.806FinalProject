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

class SimpleLossCompute:
  """A simple loss compute and train function."""

  def __init__(self, criterion, model, opt=None):
    self.criterion = criterion
    self.opt = opt
    self.model = model
    self.max_grad_norm = 1

  def __call__(self, x, y, norm, grad_verbose = False):
    x = torch.log(x) # get log-probs for use of NLL loss
    
    loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                          y.contiguous().view(-1))

    loss = loss / norm

    if self.opt is not None:  # training mode
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm) # added for numerical stability         
      self.opt.step()

      # Examine gradients for divergence (https://discuss.pytorch.org/t/how-can-i-get-the-gradients-of-the-weights-of-each-layer/28502)
      if grad_verbose:
        for param, weight in copynet.named_parameters():
          if "bias" not in param:
            print('===========\ngradient:{}\n----------\n{}'.format(param, torch.all(torch.isnan(weight.grad))))
      
      self.opt.zero_grad()

    return loss.data.item() * norm

def run_epoch(data_loader, model, loss_compute, print_every, save_every, save_name):
  """
    Standard Training and Logging Function
    -cp_lst: additional indices for copy mechanism (same length as data loader)
  """

  total_tokens = 0
  total_loss = 0

  for i, (src_ids_BxT, src_lengths_B, src_ids_cp_BxT, trg_ids_BxL, trg_lengths_B) in enumerate(data_loader):
    src_ids_BxT = src_ids_BxT.to(device)
    src_lengths_B = src_lengths_B.to(device)
    src_ids_cp_BxT = src_ids_cp_BxT.to(device)
    trg_ids_BxL = trg_ids_BxL.to(device)

    del trg_lengths_B   # unused

    output = model(src_ids_BxT, src_ids_cp_BxT, trg_ids_BxL, src_lengths_B) # copy indices shouldn't mess up trg_id evaluation because tacked on at end!

    grad_flag = False #(i % 5 == 0)
    loss = loss_compute(x=output, y=trg_ids_BxL[:, 1:], norm=src_ids_BxT.size(0), grad_verbose = grad_flag) # also does backward step; normalization is by batch

    total_loss += loss
    total_tokens += (trg_ids_BxL[:, 1:] != PAD_INDEX).data.sum().item()

    if model.training and i % print_every == 0:
      print("Epoch Step: %d Loss: %f" % (i, loss / src_ids_BxT.size(0)))
    if model.training and i % save_every == 0 and i != 0:
      print("Model saved!")
      torch.save(model.state_dict(), MODEL_FOLDER + "/" + save_name)

  return math.exp(total_loss / float(total_tokens))

def train(model, num_epochs, learning_rate, print_every, save_every, save_name):
  """Standard training function."""
  # Set `ignore_index` as PAD_INDEX so that pad tokens won't be included when
  # computing the loss.
  criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_INDEX)
  optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

  # Keep track of dev ppl for each epoch.
  dev_ppls = []

  for epoch in range(num_epochs):
    print("Epoch", epoch)

    model.train()

    start_time = time()
    train_ppl = run_epoch(data_loader=train_data_loader,
                          model=model, loss_compute=SimpleLossCompute(criterion, model, optim),
                          print_every=print_every, save_every=save_every, save_name=save_name)
    end_time = time()
    print(f"Time for Epoch {epoch}: {round((end_time - start_time) / 60., 1)} minutes")
    model.eval()
    with torch.no_grad():      
      dev_ppl = run_epoch(data_loader=val_data_loader,
                          model=model, loss_compute=SimpleLossCompute(criterion, model, None),
                          print_every=print_every, save_every=save_every, save_name=save_name)
      print("Validation perplexity: %f" % dev_ppl)
      dev_ppls.append(dev_ppl)
        
  return dev_ppls