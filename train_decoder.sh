#!/bin/sh
################################################################################
# Train a Decoder-Only Predictive Coding Transformer on Shakespeare Data
################################################################################

DATA_DIR="./Data"
TOKENIZED_DIR="$DATA_DIR/tokenized_ptb"

# Optional: Clear previous experiment logs and checkpoints
rm -rf exp/*
mkdir -p exp

# Run the training script
python train.py \
    --data_dir "$DATA_DIR" \
    --tokenized_dir "$TOKENIZED_DIR" \
    --verbosity 1
