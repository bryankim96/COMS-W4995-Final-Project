#!/bin/bash

read -p "Enter a test description: "  query
echo "Description: $query"

embedding_net="../pretrained_checkpoints/lm_sje_cub_c10_hybrid_0.00070_1_10_trainvalids.txt.t7"
echo "Using pretrained embedding net: $embedding_net"

query_str=$query \
embedding_net=$embedding_net \
th save_embeddings.lua

python ../train.py stage1 ../pretrained_checkpoints/run1 --mode=eval --test_description="$query" --test_embedding="./test_embedding.t7"

rm "./test_embedding.t7"
