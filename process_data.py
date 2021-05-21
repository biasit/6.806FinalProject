import sys
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

## Data initialization helper functions

# Take list of sentences and create the vocab
def create_vocab(data_l, vfile):
    vocab = set()
    for line in data_l:
        vocab |= set(line)
    with open(vfile, 'w') as vf:
        for v in vocab:
            vf.write(v+'\n')

# Check if valid source
def check_if_valid_source(data_src):
  for sent in data_src:
    assert sent[0] == "__User" and sent[-1] == "__StartOfProgram"

  return None

# Add in start of program and end of program tokens
def add_start_eos(data_trg):
  for i_s, sent in enumerate(data_trg):
      data_trg[i_s] = [START_TOKEN] + data_trg[i_s]
      data_trg[i_s] += [END_TOKEN]

  return data_trg

# Drop src/trg sentences longer than the above
def drop_longer(data_src, data_trg, max_src_length = MAX_SRC_LENGTH, max_trg_length = MAX_TRG_LENGTH):
  i_s = 0
  for i in range(len(data_src)):
    if (len(data_src[i_s]) > max_src_length) or (len(data_trg[i_s]) > max_trg_length):
        data_src.pop(i_s)
        data_trg.pop(i_s)
        i_s -= 1

    i_s += 1

  return data_src, data_trg

# Get maximize size
def get_dataset_length(data_src):
  dataset_len = 0
  for sent in train_data_src:
      if len(sent) > dataset_len:
          MAX_SRC_LENGTH = len(sent)

# Load in the train data and create vocabulary files (different usage for train/val)
def load_data(data_dir, data_src_path, data_trg_path, get_vocab = True):
  data_src = open(data_src_path).read().split('\n')
  data_src = [t.split() for t in data_src]
  data_trg = open(data_trg_path).read().split('\n')
  data_trg = [t.split() for t in data_trg]

  # Remove empty sentences  - may want to check this is ok
  for i_s, sent in enumerate(data_src):
      if len(sent) == 0:
          data_src.pop(i_s)
          data_trg.pop(i_s)

  if not get_vocab:
    return data_src, add_start_eos(data_trg)

  else:
    # May need to recompute the vocabulary for the smaller training set
    src_vocab_file = os.path.join(data_dir, "train.src.vocab")
    if not os.path.exists(src_vocab_file):
        create_vocab(data_src, src_vocab_file)
    src_vocab = set(open(src_vocab_file).read().split('\n'))

    trg_vocab_file = os.path.join(data_dir, "train.trg.vocab")
    if not os.path.exists(trg_vocab_file):
        create_vocab(data_trg, trg_vocab_file)
    trg_vocab = set(open(trg_vocab_file).read().split('\n'))

    return data_src, add_start_eos(data_trg), src_vocab, trg_vocab

  
## Data Class

# Create our Dataset (largely taken from PSet 3 of 6.806)
class DataflowDataset(torch.utils.data.Dataset):
    def __init__(self, src_sentences, src_vocab, trg_sentences, trg_vocab):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences

        self.max_src_seq_length = MAX_SRC_LENGTH
        self.max_trg_seq_length = MAX_TRG_LENGTH

        self.src_vocab = src_vocab | set(SAVED_INDEXES.keys())
        self.trg_vocab = trg_vocab | set(SAVED_INDEXES.keys())
        
        both_src_trg_vocab = self.src_vocab & self.trg_vocab 
        self.src_v2id = {}
        self.trg_v2id = {}
        # add in SAVED INDEXES
        for v, id in SAVED_INDEXES.items():
            self.src_v2id[v] = id
            self.trg_v2id[v] = id

        # add in vocab in both vocabularies
        for v in sorted(list(both_src_trg_vocab - set(SAVED_INDEXES.keys()))):
            self.src_v2id[v] = len(self.src_v2id)
            self.trg_v2id[v] = len(self.trg_v2id)
          
        # finish up src vocab
        for v in sorted(list(self.src_vocab - both_src_trg_vocab)):
            self.src_v2id[v] = len(self.src_v2id)
        
        # finish up trg vocab
        for v in sorted(list(self.trg_vocab - both_src_trg_vocab)):
            self.trg_v2id[v] = len(self.trg_v2id)
        
        self.src_id2v = {val : key for key, val in self.src_v2id.items()}
        self.trg_id2v = {val : key for key, val in self.trg_v2id.items()}

        # preprocess the dataset...
        self.preprocess_dataset()

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        return self.src_ids[index], self.src_lens[index], self.src_copy_ids[index], \
               self.trg_ids[index], self.trg_lens[index]


    def preprocess_item(self, index):
        src_len = len(self.src_sentences[index])
        trg_len = len(self.trg_sentences[index])
        src_id = self.src_idx(index)
        
        src_copy_idx, trg_id = self.copy_idx(index)

        return torch.tensor(src_id), src_len, torch.tensor(src_copy_idx), torch.tensor(trg_id), trg_len
    
    def src_idx(self, index):
        src_sent = self.src_sentences[index]
        src_len = len(src_sent)

        src_id = []
        for w in src_sent:
            if w not in self.src_vocab:
                w = UNK_TOKEN
            src_id.append(self.src_v2id[w])
        src_id += [PAD_INDEX] * (self.max_src_seq_length - src_len)
        return src_id
    
    def convert_from_copy_trg_vocab_to_orig(self, index, sentence):
        """
            Sentence: target sentence ids
            If you have a sentence that is a list of target ids and want the 
            original target tokens, use this function.
            Index should be the corresponding index for the instance level
            vocab that you will use.
        """
        new_term_dict = {}
        src_sent = self.src_sentences[index]

        next_idx = len(self.trg_vocab)
        for w in src_sent:
            if w not in self.trg_vocab and w not in new_term_dict:
                new_term_dict[w] = next_idx 
                next_idx += 1
        rev_new_term_dict = {val: key for key, val in new_term_dict.items()} # id to v

        ret_sent = []
        for id in sentence:
            if id in self.trg_id2v:
                ret_sent.append(self.trg_id2v[id])
            elif id in rev_new_term_dict:
                ret_sent.append(rev_new_term_dict[id])
            else:
                ret_sent.append(UNK_TOKEN)
        return ret_sent
    
    def convert_from_src_sent_to_input(self, src_sent):
        src_len = len(src_sent)

        src_id = []
        for w in src_sent:
            if w not in self.src_vocab:
                w = UNK_TOKEN
            src_id.append(self.src_v2id[w])
        src_id += [PAD_INDEX] * (self.max_src_seq_length - src_len)

        # Get src_ids with expanded vocabulary
        src_cp_id = []
        new_term_dict = {}
        next_idx = len(self.trg_vocab)
        for w in src_sent:
            if w in self.trg_vocab:
                src_cp_id.append(self.trg_v2id[w])
            else:
                if w in new_term_dict:
                    src_cp_id.append(new_term_dict[w])
                else:
                    new_term_dict[w] = next_idx
                    src_cp_id.append(next_idx)
                    next_idx += 1
        src_cp_id += [PAD_INDEX] * (self.max_src_seq_length - src_len)

        return torch.tensor(src_id), src_len, torch.tensor(src_cp_id)

    def convert_from_copy_trg_vocab_with_src(self, src_sent, sentence):
        """
            Src Sent: original src sentence from which expanded vocabulary is created
            Sentence: target sentence ids
            If you have a sentence that is a list of target ids and want the 
            original target tokens, use this function.
            Src_sent should be the corresponding source sentence from which
            copy indices were generated. This should be tokenized list!
        """
        new_term_dict = {}

        next_idx = len(self.trg_vocab)
        for w in src_sent:
            if w not in self.trg_vocab and w not in new_term_dict:
                new_term_dict[w] = next_idx 
                next_idx += 1
        rev_new_term_dict = {val: key for key, val in new_term_dict.items()} # id to v

        ret_sent = []
        for id in sentence:
            if id in self.trg_id2v:
                ret_sent.append(self.trg_id2v[id])
            elif id in rev_new_term_dict:
                ret_sent.append(rev_new_term_dict[id])
            else:
                ret_sent.append(UNK_TOKEN)
        return ret_sent

    def copy_idx(self, index):
        """
            Generate indices based on expanded vocabulary for copying. 
        """
        src_sent = self.src_sentences[index]
        src_len = len(src_sent)

        trg_sent = self.trg_sentences[index]
        trg_len = len(trg_sent)

        # Get src_ids with expanded vocabulary
        src_id = []
        new_term_dict = {}
        next_idx = len(self.trg_vocab)
        for w in src_sent:
            if w in self.trg_vocab:
                src_id.append(self.trg_v2id[w])
            else:
                if w in new_term_dict:
                    src_id.append(new_term_dict[w])
                else:
                    new_term_dict[w] = next_idx
                    src_id.append(next_idx)
                    next_idx += 1
        src_id += [PAD_INDEX] * (self.max_src_seq_length - src_len)

        # Get trg_ids with expanded vocabulary
        trg_id = []
        for w in trg_sent:
            if w in self.trg_vocab:
                trg_id.append(self.trg_v2id[w])
            else:
                if w in new_term_dict:
                    trg_id.append(new_term_dict[w])
                else:
                    w = UNK_TOKEN
                    trg_id.append(self.trg_v2id[w]) 
        trg_id += [PAD_INDEX] * (self.max_trg_seq_length - trg_len)

        return src_id, trg_id

    def preprocess_dataset(self):
        all_src = []
        all_src_lengths = []
        all_src_copy_ids = []
        all_trg = []
        all_trg_lengths = []
        for i in range(len(self)):
            curr_src, curr_src_len, curr_src_copy_id, curr_trg, curr_trg_len = self.preprocess_item(i)
            all_src.append(curr_src[None,:])
            all_src_lengths.append(curr_src_len)
            all_src_copy_ids.append(curr_src_copy_id[None,:])
            all_trg.append(curr_trg[None,:])
            all_trg_lengths.append(curr_trg_len)

        self.src_ids = torch.cat(all_src)
        self.src_lens = all_src_lengths 
        self.src_copy_ids = torch.cat(all_src_copy_ids)
        self.trg_ids = torch.cat(all_trg)
        self.trg_lens = all_trg_lengths

# Load in dataflow
def load_dataflow(dataset_path, data_src, data_trg, src_vocab, trg_vocab):
  if not os.path.exists(dataset_path):
    dataset = DataflowDataset(data_src, src_vocab,
                                data_trg, trg_vocab)
    with open(dataset_path,'wb') as f:
        pickle.dump(dataset, f)

  else:
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

  return dataset