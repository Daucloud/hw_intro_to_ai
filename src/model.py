import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx

PAD_IDX = 0

class BaseModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=PAD_IDX,
                 pretrained_embedding=None, freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        if pretrained_embedding is not None:
            print("Loading pre-trained weights into embedding layer...")
            pretrained_tensor = torch.tensor(pretrained_embedding, dtype=torch.float)
            if pretrained_tensor.shape == self.embedding.weight.data.shape:
                self.embedding.weight.data.copy_(pretrained_tensor)
            else:
                 print(f"Warning: Shape mismatch! Pretrained weights shape {pretrained_tensor.shape} "
                       f"does not match Embedding layer shape {self.embedding.weight.data.shape}. "
                       f"NOT loading weights.")

            self.embedding.weight.requires_grad = not freeze_embeddings
            if freeze_embeddings:
                print("Embedding layer weights frozen.")
            else:
                print("Embedding layer weights will be fine-tuned.")
        else:
            print("No pre-trained weights provided, using random initialization.")

class CNN(BaseModel):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim,
                 dropout=0.5, pretrained_embedding=None, freeze_embeddings=True):
        super().__init__(vocab_size, embedding_dim, PAD_IDX, pretrained_embedding, freeze_embeddings)

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        conved = [c.squeeze(3) for c in conved]
        pooled = [F.max_pool1d(c, kernel_size=c.shape[2]) for c in conved]
        pooled = [p.squeeze(2) for p in pooled]
        cat = torch.cat(pooled, dim=1)
        dropped = self.dropout(cat)
        return self.fc(dropped)

class RNN(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 num_layers=2, bidirectional=True,
                 dropout=0.5, pretrained_embedding=None, freeze_embeddings=True):
        super().__init__(vocab_size, embedding_dim, PAD_IDX, pretrained_embedding, freeze_embeddings)

        gru_dropout = dropout if num_layers > 1 else 0
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True,
                          dropout=gru_dropout, bidirectional=bidirectional, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.gru(embedded)
        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        return self.fc(hidden)

class MLP(BaseModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 dropout=0.5, pretrained_embedding=None, freeze_embeddings=True):
        super().__init__(vocab_size, embedding_dim, PAD_IDX, pretrained_embedding, freeze_embeddings)

        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)

        mask = (text != PAD_IDX).unsqueeze(-1).float()
        masked_embedded = embedded * mask
        summed = masked_embedded.sum(dim=1)
        non_pad_count = mask.sum(dim=1).clamp(min=1e-6)
        sentence_embedding = summed / non_pad_count

        hidden = self.dropout(F.relu(self.fc1(sentence_embedding)))
        return self.fc2(hidden)