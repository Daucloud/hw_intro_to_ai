#! /bin/bash

model_type=$1
epochs=$2
batch_size=$3
lr=$4

python src/main.py \
    --model_type "$model_type" \
    --data_dir Dataset \
    --embedding_file wiki_word2vec_50.bin \
    --num_epochs "$epochs" \
    --batch_size "$batch_size" \
    --learning_rate "$lr" \
    --dropout 0.5 \
    --embedding_dim 50 \
    --max_len 128 \
    --freeze_embeddings True \
    --early_stopping_patience 10 \
    --seed 42 \
    --cnn_num_filters 1000 \
    --cnn_filter_sizes 3,4,5 \
    --rnn_hidden_dim 128 \
    --rnn_num_layers 3 \
    --rnn_bidirectional True \
    --rnn_type GRU \
    --mlp_hidden_dim 64 \
    --wandb_project Intro-to-AI-Assignment2 \
    --wandb_disabled