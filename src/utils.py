from gensim.models import KeyedVectors
import numpy as np
from torch.utils.data import Dataset
import torch

PAD="<PAD>"
UNK="<UNK>"

def build_vocab(path="Dataset/train.txt"):
    vocab = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            vocab.update(words[1:])
    word2idx = {PAD: 0, UNK: 1}
    for i, word in enumerate(vocab):
        word2idx[word] = i + 2
    idx2word = {0: PAD, 1: UNK}
    for i, word in enumerate(vocab):
        idx2word[i + 2] = word
    return vocab, word2idx, idx2word

def load_word2vec(path="Dataset/wiki_word2vec_50.bin"):
    word2vec = KeyedVectors.load_word2vec_format(path, binary=True)
    return word2vec

def build_embedding(word2idx, word2vec):
    embedding = np.zeros((len(word2idx), word2vec.vector_size))
    for word, idx in word2idx.items():
        if word in word2vec:
            embedding[idx] = word2vec[word]
    embedding[1] = np.random.normal(0, 0.01, word2vec.vector_size)
    return embedding

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return [(int(words[0]), words[1:]) for line in f for words in [line.strip().split()]]

class SentimentDataset(Dataset):
    def __init__(self, data, word2idx, max_len):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, words = self.data[idx]
        words = [self.word2idx.get(word, self.word2idx[UNK]) for word in words]
        words = words[:self.max_len]
        words = words + [self.word2idx[PAD]] * (self.max_len - len(words))
        return torch.LongTensor(words), torch.LongTensor(label)