import numpy as np
import os
from ..global_variables import *

def get_embeddings(src_sentences):
    '''!pip install -U sentence-transformers'''
    # Download the pretrained model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    return model.encode(src_sentences)

def embed_sentences(embed_path, sentences):
    '''Use SentenceBert to embed sentences'''
    if not os.path.exists(embed_path):
        # Get train embeddings
        src = [' '.join(sent) for sent in sentences]
        embeddings = get_embeddings(src)
        # Save train_embeddings
        with open(embed_path,'wb') as f:
            np.save(f, embeddings)
    else:
        with open(embed_path,'rb') as f:
            embeddings = np.load(f)
    return embeddings

def get_most_similar_trg_sentences(train_embeddings, embeddings, from_src=None):
    '''Find indexes of most similar train set sentences to embedded
        sentences in embeddings'''
    from tqdm import tqdm
    tr_embeddings_norm = np.expand_dims(np.linalg.norm(train_embeddings, axis=1),1)

    # similarity coeffs based on cosine distance
    best_indexes = np.zeros(len(embeddings))

    batch_s = 5000
    for i in tqdm(range(0,len(embeddings),batch_s)):
      curr_sim = np.matmul(train_embeddings, embeddings[i:i+batch_s].T)/np.matmul(tr_embeddings_norm, np.linalg.norm(embeddings[i:i+batch_s],axis=1)[None,:])
      if from_src:
          for j in range(i,min(i+batch_s,len(embeddings))):
              curr_sim[j,j-i] = float('-inf')
      best_indexes[i:i+batch_s] = np.argmax(curr_sim, axis=0)
    return best_indexes

def append_retrieved_target(train_src, train_trg, curr_src, from_train=True):
    ''' Get the retrieved sentences and append to curr_src '''
    train_embed_path = './model_input/train_embeddings_length200.np'
    if from_train:
        embed_path = train_embed_path
    else:
        embed_path = './model_input/val_embeddings_length200.np'

    train_embeddings = embed_sentences(train_embed_path, train_src)
    curr_embeddings = embed_sentences(embed_path, curr_src)

    # get ids of sentences with highest similarity
    sim_ids = get_most_similar_trg_sentences(train_embeddings, curr_embeddings, from_src=from_train)

    # modify the src sentences
    for idx, sim_idx in enumerate(sim_ids):
        # get the retrieved sentence
        retrieved_sent = train_trg[int(sim_idx)]
        orig_sent = curr_src[idx]
        combined_sent = orig_sent+[RETR_TOKEN]+retrieved_sent
        curr_src[idx] = combined_sent
    return curr_src
