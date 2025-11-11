from jax import numpy as jnp, random
import time, os
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
from Dataloader import BPETokenizer, PennTreeBankDataset, create_dataloader
from ngclearn.utils.metric_utils import measure_CatNLL
import numpy as np
from EmbeddingRateCell import Embedding
from ngcsimlib.context import Context
import jax
from jax import random as jax_random
from pcn import PCN
# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 16
BLOCK_SIZE = 128
N_ITER = 2
LEARNING_RATE = 1e-3
SAVE_POINT = 1
VERBOSITY = 1
n_embed=512
dim=n_embed

# -----------------------------
# Tokenizer and Data
# -----------------------------
bpe = BPETokenizer()
vocab_size = bpe.vocab_size
pad_id = bpe.pad_token_id

train_dataset = PennTreeBankDataset("train_ids.pkl", "./Data/tokenized_ptb", BLOCK_SIZE)
val_dataset   = PennTreeBankDataset("valid_ids.pkl", "./Data/tokenized_ptb", BLOCK_SIZE)

train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True, pad_token_id=pad_id)
val_loader   = create_dataloader(val_dataset, BATCH_SIZE, shuffle=False, pad_token_id=pad_id)

# -----------------------------
# Model setup
# -----------------------------

dkey = random.PRNGKey(1234)   # shape (2,)

model=PCN(dkey=dkey,dim=dim,T=10,dt=1.,tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", batch_size=BATCH_SIZE,loadDir=None)
print("model run ")

# -----------------------------
# Training / Evaluation
# -----------------------------
def compute_loss(logits, targets, pad_token_id=0):
    """
    Compute negative log-likelihood ignoring padding.
    logits: [batch, seq_len, vocab_size]
    targets: [batch, seq_len]
    """
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    nll = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
    mask = (targets != pad_token_id)
    nll = jnp.sum(nll * mask) / jnp.sum(mask)
    return nll

# -----------------------------
# Main Training Loop
# -----------------------------
print("--- Starting Training ---")
sim_start_time = time.time()

for epoch in range(N_ITER):
    total_loss = 0
    n_batches = 0

    for batch in train_loader:
        Xb, Yb = batch["input_ids"], batch["target_ids"]
        
    # Show first 10 tokens of first sample
        
        # Use the pre-created embedding cell
        E.j.set(Xb)           # Set the input tokens
        result=E.advance_state()     # Compute embeddings
        embeddings = result  # Get output of shape (batch_size, seq_len, 512)

        
        
        print(f"Embedding shape: {embeddings.shape}")
        print("First sample embedding stats:")
        # print(embeddings)
        break
        
        # Show actual embedding values for first token
        # if embeddings[0, 0, 0] != 0:  # If not zero, show some values
        #     print("First few embedding values for first token:")
        #     print(embeddings[0, 0, :8])  # First 8 dimensions

        # # Reset for next batch (optional - depends on your use case)
        # # E.reset()

        # n_batches += 1

        # if VERBOSITY >= 1 and n_batches % 10 == 0:
        #     print(f"Epoch {epoch} | Batch {n_batches} | Processed {Xb.shape} inputs")

        # if n_batches >= 2:  # Just test with 2 batches for now
        #     break

    break  # Just run one epoch for testing

sim_time = time.time() - sim_start_time
print(f"Training completed in {sim_time/60:.2f} min.")