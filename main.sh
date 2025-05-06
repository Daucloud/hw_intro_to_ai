#! /bin/bash

python src/main.py \
    --model_type rnn \
    --data_dir Dataset \
    --embedding_file wiki_word2vec_50.bin \
    --num_epochs 10 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --dropout 0.5 \
    --embedding_dim 50 \
    --max_len 128 \
    --freeze_embeddings True \
    --early_stopping_patience 5 \
    --seed 42 \
    --cnn_num_filters 100 \
    --cnn_filter_sizes 3,4,5 \
    --rnn_hidden_dim 128 \
    --rnn_num_layers 1 \
    --rnn_bidirectional True \
    --rnn_type GRU \
    --mlp_hidden_dim 64 \
    --wandb_project Intro-to-AI \
    --wandb_notes assignment2 \
    --wandb_disabled