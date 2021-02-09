# encoding:utf-8
import os
import ujson
from torchtext import data
from transformers import BertTokenizer
import pickle
import numpy as np
import logging

logging.basicConfig(filename='preProcessor.log', level=logging.INFO)
def pkl_read(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def pkl_write(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


class GeneralDataset(data.Dataset):
    def __init__(self, file_path):
        self.input = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    js = ujson.loads(line.strip())
                except:
                    js = ujson.loads(line.strip().replace('\'', '\"'))
                self.input.append(js)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return (self.input[index])

    def save(self):
        pass

    def load(self):
        pass


class GeneralTransformer(object):
    def __init__(self, vocab_path, embedding_path, bert_tokenizer=None):
        if os.path.isfile(vocab_path):
            self.vocab = pkl_read(vocab_path)
            self.rev_vocab = {v: k for k, v in self.vocab.items()}
        if os.path.isfile(embedding_path):
            embeddings_index = {}
            try:
                logging.info("loading_embedding_matrix...")
                self.embedding_matrix = np.load("emb.npy")
                logging.info("loaded.")
            except:
                logging.error("no embedding matrix found")
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
                            logging.error("Error on ", values[:2])

                all_embs = np.stack((embeddings_index.values()))
                emb_mean = all_embs.mean()
                emb_std = all_embs.std()
                embed_size = all_embs.shape[1]
                nb_words = len(self.vocab)

                embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
                for word, id in self.vocab.items():
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        embedding_matrix[id] = embedding_vector
                self.embedding_matrix = embedding_matrix.astype(np.float32)

                np.save("emb.npy", self.embedding_matrix)
        if bert_tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        else:
            pass
