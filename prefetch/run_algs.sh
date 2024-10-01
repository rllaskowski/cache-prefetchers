#!/bin/bash

histories=1
if [[ "$1" == "cyclic"* ]]; then
  setup="--$1 --cache_size=2 --trace_size=1000"
  n=6
elif [[ "$1" == "markov_chain" ]]; then
  setup="--$1 --cache_size=2 --trace_size=1000"
  n=8
else
  setup="--trace_path=/Users/robertlaskowski/Desktop/studia/projects/prefetching/trace/Trace/11 --cache_size=16"
  histories="14"
  n=64
fi
main="python main.py $setup"

for algo in size opt random lru mq lru_next_prefetch mq_next_prefetch; do
  $main --algo=$algo | tail -1
done



for sps in 4; do
  for lambda in 1e-3; do
    for eta in 0.1; do
      for h in $histories; do
        $main --n=$n --k=4 --history_size=$h --samples_per_step=$sps --lambda=$lambda --eta=$eta --algo=dom_ffm | tail -1
        #$main --n=$n --k=4 --history_size=$h --samples_per_step=$sps --lambda=$lambda --eta=$eta --algo=dom_ffm_next_prefetch | tail -1
        #for p in 0.05; do
        #  $main --n=$n --k=4 --history_size=$h --samples_per_step=$sps --lambda=$lambda --eta=$eta --prob_threshold=$p --algo=dom_ffm_mixed_prefetch | tail -1
        #done
      done
    done
  done
done
