# train_pcn_lm.py
import time
import os
from jax import random
import jax
import jax.numpy as jnp
import numpy as np

from Dataloader import BPETokenizer, PennTreeBankDataset, create_dataloader
from ngclearn.utils.metric_utils import measure_CatNLL
from pcn import PCN

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 16
BLOCK_SIZE = 128
N_EPOCHS = 2         # change to something higher for real training
SAVE_POINT = 1
VERBOSITY = 1
LEARNING_RATE = 1e-3
SEED = 1234
MAX_NEW_TOKENS = 50
n_embed=32


n_heads=8
dropout_rate=0.5

# -----------------------------
# Tokenizer and Data
# -----------------------------
bpe = BPETokenizer()
vocab_size = bpe.vocab_size
pad_id = bpe.pad_token_id
eos_token_id = bpe.tokenizer.eot_token if hasattr(bpe, "tokenizer") else None

# update paths if necessary
DATA_DIR = "./Data/tokenized_ptb"
train_dataset = PennTreeBankDataset("train_ids.pkl", DATA_DIR, BLOCK_SIZE)
val_dataset   = PennTreeBankDataset("valid_ids.pkl", DATA_DIR, BLOCK_SIZE)

train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True, pad_token_id=pad_id)
val_loader   = create_dataloader(val_dataset, BATCH_SIZE, shuffle=False, pad_token_id=pad_id)

# -----------------------------
# Model setup
# -----------------------------
dkey = random.PRNGKey(SEED)
dim = 512   # model embedding dim (match your PCN instantiation)
model = PCN(dkey=dkey, dim=dim, T=10, dt=1., tau_m=10., act_fx="tanh",
            eta=0.001, exp_dir="exp", model_name="pcn_lm", block_size=BLOCK_SIZE,n_embed=n_embed,batch_size=BATCH_SIZE,n_heads=n_heads,dropout_rate=dropout_rate, loadDir=None)
print("Model built")

# -----------------------------
# Utilities
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

def eval_model(model, loader, pad_token_id=0):
    """Evaluate average NLL on loader (no adaptation of synapses)."""
    total_nll = 0.0
    n_batches = 0
    for batch in loader:
        Xb = batch["input_ids"]    # jnp array [B, L]
        Yb = batch["target_ids"]
        # Run model in inference mode (no synaptic adaptation)
        # ASSUMPTION: model.process(obs, lab, adapt_synapses=False) returns (yMu_0, yMu, EFE)
        yMu_0, yMu, _ = model.process(obs=Xb, lab=Yb, adapt_synapses=False)
        nll = compute_loss(yMu_0, Yb, pad_token_id=pad_token_id)
        total_nll += float(nll)
        n_batches += 1
    return total_nll / max(1, n_batches)

# -----------------------------
# Training loop
# -----------------------------
print("--- Starting Training ---")
start_time = time.time()

for epoch in range(N_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0

    # shuffle is handled by create_dataloader generator if shuffle=True
    for batch in train_loader:
        Xb = batch["input_ids"]
        Yb = batch["target_ids"]

        # NOTE: PCN process typically does inference + learning when adapt_synapses=True
        # ASSUMPTION: returns (yMu_0, yMu, EFE) where yMu_0 are logits/predictions
        yMu_0, yMu, efe = model.process(obs=Xb, lab=Yb, adapt_synapses=True)

        # compute NLL for monitoring
        batch_nll = float(compute_loss(yMu_0, Yb, pad_token_id=pad_id))
        epoch_loss += batch_nll
        n_batches += 1

        if VERBOSITY and (n_batches % 50 == 0):
            print(f"Epoch {epoch} | Batch {n_batches} | batch_nll={batch_nll:.4f} | efe={efe:.4f}")

    avg_epoch_loss = epoch_loss / max(1, n_batches)
    print(f"Epoch {epoch} finished. Avg NLL = {avg_epoch_loss:.4f}")

    # validation
    val_nll = eval_model(model, val_loader, pad_token_id=pad_id)
    print(f"Validation NLL after epoch {epoch}: {val_nll:.4f}")

    # save model periodically (assumes model.save_to_disk exists and supports params_only)
    if (epoch + 1) % SAVE_POINT == 0:
        try:
            model.save_to_disk(params_only=True)
            print("Model saved to disk.")
        except Exception as e:
            print("Warning: model.save_to_disk failed:", e)

end_time = time.time()
print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes")

# -----------------------------
# Generation helper
# -----------------------------
def generate(model, seed_ids, max_new_tokens=50, temperature=1.0, eos_token_id=None, block_size=BLOCK_SIZE, rng_seed=SEED):
    """
    Generate new tokens autoregressively using the trained PCN.
    seed_ids: 1D array-like of token ids (seed prompt)
    Returns the generated token array (including seed).
    """
    # ensure jnp arrays with batch dimension
    seq = list(map(int, seed_ids))
    key = random.PRNGKey(rng_seed)

    for _ in range(max_new_tokens):
        # build input tensor with batch dim 1
        cur = jnp.array([seq], dtype=jnp.int32)  # shape (1, L)
        # trim to block_size last tokens
        if cur.shape[1] > block_size:
            cur = cur[:, -block_size:]

        # call model in inference mode: do not adapt synapses
        # ASSUMPTION: model.process(obs, lab, adapt_synapses=False) returns (yMu_0, yMu, EFE)
        # yMu_0 expected shape [batch, seq_len, vocab_size]
        yMu_0, yMu, _ = model.process(obs=cur, lab=jnp.zeros_like(cur), adapt_synapses=False)

        # take last token logits
        last_logits = jnp.asarray(yMu_0)[0, -1, :]   # shape [vocab_size]

        # scale by temperature
        scaled = last_logits / float(max(1e-8, temperature))

        # sample next token
        key, sub = random.split(key)
        next_token = int(random.categorical(sub, scaled, axis=-1))

        seq.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return jnp.array(seq, dtype=jnp.int32)

# # -----------------------------
# # Example generate after training
# # -----------------------------
# # pick first example from validation data as seed
# for batch in val_loader:
#     seed = batch["input_ids"][0]  # jnp 1D/possibly padded
#     # trim zeros/pads to get a shorter seed if needed (optional)
#     seed_trimmed = seed[:64]      # use first 64 tokens of validation sample as seed
#     generated = generate(model, seed_trimmed, max_new_tokens=50, temperature=1.0, eos_token_id=eos_token_id)
#     print("Seed tokens:", seed_trimmed)
#     print("Generated token ids:", generated)
#     break

# # optionally convert token ids back to text using your BPE tokenizer
# try:
#     decoded = bpe.tokenizer.decode(list(map(int, generated.tolist())))
#     print("Generated text:", decoded)
# except Exception:
#     print("Could not decode generated IDs to text with current tokenizer.")
