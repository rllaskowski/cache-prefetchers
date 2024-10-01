srun --qos 1gpu1d --gres=gpu:1 python embeding_lstm.py \
    --num_epochs 15 \
    --seq_len 64 \
    --vocab_in ./vocabs/e_lstm/vocab_in_th_2.json \
    --vocab_in ./vocabs/e_lstm/vocab_out_percent_10.json \
    --dropout 0.8 \
    --wd 0.05 \
    --hidden_size 128 \
    --embedding_size 128 \
    --dir ./experiments/e_lstm/seq_len_64_vocab_in_th_2_vocab_out_percent_15_dropout_08_embedding_128_hidden_64
