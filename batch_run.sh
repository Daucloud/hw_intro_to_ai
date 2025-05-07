#! /bin/bash

for model_type in cnn rnn mlp; do
    for epochs in 10 100; do
        for batch_size in 32 64 128; do
            for lr in 0.001 0.0001 0.00001; do
                bash main.sh $model_type $epochs $batch_size $lr
            done
        done    
    done
done
