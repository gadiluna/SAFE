#!/bin/sh


BASE_PATH="/home/luca/work/binary_similarity_data/"

DATA_PATH=$BASE_PATH/experiments/arith_mean_openSSL_no_dropout_no_shuffle_no_regeneration_emb_random_trainable
OUT_PATH=$DATA_PATH/out

DB_PATH=$BASE_PATH/databases/openSSL_data.db

EMBEDDER=$BASE_PATH/word2vec/filtered_100_embeddings/

RANDOM=""
TRAINABLE_EMBEDD=""

python3 train.py $RANDOM $TRAINABLE_EMBEDD --o $OUT_PATH -n $DB_PATH -e $EMBEDDER
