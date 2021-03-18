# encoding:utf-8
import os
from transformers import BertTokenizer
import pickle
import numpy as np
import logging

logger = logging.Logger(__name__)

def pkl_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pkl_write(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)



class GeneralTransformer(object):
    EMBEDDING_MATRIX_MEM = None

    def __init__(self, vocab_path, embedding_path, bert_tokenizer=None):
        if os.path.isfile(vocab_path):
            self.vocab = pkl_read(vocab_path)
            self.rev_vocab = {v: k for k, v in self.vocab.items()}
        if os.path.isfile(embedding_path):
            if GeneralTransformer.EMBEDDING_MATRIX_MEM is None:
                embeddings_index = {}
                with open(embedding_path, 'r', errors='ignore', encoding='utf8') as f:
                    for line in f:
                        values = line.rstrip().split(' ')

                        if len(values) < 10:
                            continue
                        try:
                            word = values[0]
                            coefs = np.asarray(values[1:], dtype='float32')
                            embeddings_index[word] = coefs
                        except:
                            logger.error("Error on ", values[:2])

                all_embs = np.stack(list(embeddings_index.values()))
                emb_mean = all_embs.mean()
                emb_std = all_embs.std()
                embed_size = all_embs.shape[1]
                nb_words = len(self.vocab)

                embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
                for word, id in self.vocab.items():
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[id] = embedding_vector
                GeneralTransformer.EMBEDDING_MATRIX_MEM = embedding_matrix.astype(np.float32)
                
            self.embedding_matrix = GeneralTransformer.EMBEDDING_MATRIX_MEM
        if bert_tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        else:
            pass
