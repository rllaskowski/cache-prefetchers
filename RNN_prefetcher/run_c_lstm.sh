srun --qos 1gpu1d --gres=gpu:1 python clustering_lstm.py \
    --num_epochs 15 \
    --seq_len 64 \
    --dropout 0.8 \
    --hidden_size 128 \
    --embedding_size 128 \
    --dir ./experiments/c_lstm/num_epochs_15_dropout_08
