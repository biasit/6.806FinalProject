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

# This requires a lot of RAM to run
def create_glove_embeddings():
    # Build GloVe dictionary (smaller size for speed rn - easy to change)
    glove_path = "./output/glove_840b/glove.840B.300d.txt" # will need to install with auxiliary_code/get_glove.sh
    glove_dict = {}

    with open(glove_path, "rb") as glove_f:
      for i_l, l in enumerate(glove_f):
        if i_l % 10 ** 5 == 0:
          print(i_l)

        line = l.decode("utf8").strip().split(" ")
        curr_w = line[0]
        curr_emb = np.array(line[1:]).astype(np.float)
        glove_dict[curr_w] = curr_emb
    return glove_dict
def split_camelcase(wd):
  wd_split = re.sub( r"([A-Z])", r" \1", wd).split()
  return [subwd.lower() for subwd in wd_split]
# Initialize GloVe embeddings for a vocabulary
def glove_init(vocab, v2id):
  glove_dict = create_glove_embeddings()
  embedding = np.random.normal(scale=0.6, size=(len(vocab), GLOVE_DIM))
  fail_match = 0

  for wd in vocab:
    wd_token = v2id[wd]

    # Normal embedding
    if wd.lower() in glove_dict.keys():
      embedding[wd_token, :] = glove_dict[wd.lower()]

    # Embed :(command)
    elif wd.startswith(":") and (wd[1:].lower() in glove_dict.keys()):
      embedding[wd_token, :] = glove_dict[wd[1:].lower()]

    # Embed camelcase functions with average
    elif all([subwd in glove_dict.keys() for subwd in split_camelcase(wd)]):
      glove_sum = np.zeros(GLOVE_DIM)
      split_lst = split_camelcase(wd)

      for subwd in split_lst:
        glove_sum += glove_dict[subwd]

      embedding[wd_token, :] = glove_sum / max(1, len(split_lst))

    else:
      fail_match += 1

  return embedding, fail_match / len(vocab)

def get_glove_embeddings(train_set):
    ''' Get glove embeddings for vocabulary '''
    data_dir='./model_input'
    src_embedding_path = os.path.join(data_dir, "src_embedding.pkl")
    trg_embedding_path = os.path.join(data_dir, "trg_embedding.pkl")

    if not os.path.exists(src_embedding_path):
        src_embed, src_mismatch = glove_init(train_set.src_vocab, train_set.src_v2id)
        with open(src_embedding_path,"wb") as f:
            pickle.dump(src_embed, f)
    else:
        with open(src_embedding_path,"rb") as f:
            src_embed = pickle.load(f)

    if not os.path.exists(trg_embedding_path):
        trg_embed, trg_mismatch = glove_init(train_set.trg_vocab, train_set.trg_v2id)
        with open(trg_embedding_path,"wb") as f:
            pickle.dump(trg_embed, f)
    else:
        with open(trg_embedding_path,"rb") as f:
            trg_embed = pickle.load(f)

    return src_embed, trg_embed

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
    src_ids_BxT = src_ids_BxT.to(DEVICE)
    src_lengths_B = src_lengths_B.to(DEVICE)
    src_ids_cp_BxT = src_ids_cp_BxT.to(DEVICE)
    trg_ids_BxL = trg_ids_BxL.to(DEVICE)

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

def train(model, train_data_loader, val_data_loader, num_epochs, learning_rate, print_every, save_every, save_name):
  """Standard training function (modeled off of 6.806 HW3)."""
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

def setup_and_start_training(model,train_set,val_set, model_save_name):

    # Get GloVe embeddings
    src_embed, trg_embed = get_glove_embeddings(train_set)

    # Set GloVe embedding as first layer of both encoder and decoder (https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3)
    pretrained_layers = ["encoder.src_embed.weight", "decoder.trg_embed.weight"]
    pretrained_dict = {"encoder.src_embed.weight": torch.tensor(src_embed), "decoder.trg_embed.weight": torch.tensor(trg_embed)}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Freezing Glove layers  - https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088
    layer_lst = list(model.state_dict().keys())
    for i, param in enumerate(model.parameters()):
      if layer_lst[i] in pretrained_layers:
        param.requires_grad = False

    train_data_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                    num_workers=1, shuffle=True)
    val_data_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=1,
                                      shuffle=False)
    train(model, train_data_loader, val_data_loader, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
            print_every=50, save_every=600, save_name=model_save_name)
    torch.save(model.state_dict(), MODEL_FOLDER + "/" + model_save_name)

def load_model_from_save(model, model_save_name):
    if DEVICE == "cuda":
        model.load_state_dict(torch.load(MODEL_FOLDER + "/" + model_save_name))
    else:
        model.load_state_dict(torch.load(MODEL_FOLDER + "/" + model_save_name, map_location=torch.device('cpu')))
