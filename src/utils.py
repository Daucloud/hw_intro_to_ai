from gensim.models import KeyedVectors
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

PAD="<PAD>"
UNK="<UNK>"

def build_word2idx(path="Dataset/train.txt"):
    word2idx = {PAD: 0, UNK: 1}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            for word in words[1:]:
                if word not in word2idx:
                    word2idx[word] = len(word2idx)
    return word2idx

def load_word2vec(path="Dataset/wiki_word2vec_50.bin"):
    return KeyedVectors.load_word2vec_format(path, binary=True)

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
    def __init__(self, data_path, word2idx, max_len):
        self.data = load_data(data_path)
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label_int, words_raw = self.data[idx]
        label = int(label_int)

        words = [self.word2idx.get(word, self.word2idx.get(UNK, 1)) for word in words_raw]
        words = words[:self.max_len]
        padded_indices = words + [self.word2idx.get(PAD, 0)] * (self.max_len - len(words))

        sequence_tensor = torch.tensor(padded_indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)
        return sequence_tensor, label_tensor