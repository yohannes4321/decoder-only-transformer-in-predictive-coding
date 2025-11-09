from jax import numpy as jnp, random
import time, os
from Dataloader import BPETokenizer, PennTreeBankDataset, create_dataloader
from pcn import PCN  # or your NGC/decoder transformer model
from ngclearn.utils.metric_utils import measure_CatNLL
import numpy as np

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 16
BLOCK_SIZE = 128
N_ITER = 2
LEARNING_RATE = 1e-3
SAVE_POINT = 1
VERBOSITY = 1

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
dkey = random.PRNGKey(1234)
dkey, subkey = random.split(dkey)

x_dim = BLOCK_SIZE
y_dim = vocab_size

print(f"Building Decoder-only Transformer: vocab={vocab_size}, block={BLOCK_SIZE}")
model = PCN(
    subkey,
    x_dim, y_dim,
    hid1_dim=64, hid2_dim=64,
    T=2, dt=1., tau_m=25.,
    act_fx="sigmoid", eta=LEARNING_RATE,
    exp_dir="exp", model_name="decoder_only_pcn"
)
model.save_to_disk()
print("--- Model initialized and saved ---")

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

def evaluate(model, loader):
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        Xb, Yb = batch["input_ids"], batch["target_ids"]
        logits, _, _ = model.process(obs=Xb, lab=None, adapt_synapses=False)
        loss = compute_loss(logits, Yb)
        total_loss += loss * Xb.shape[0]
        total_tokens += Xb.shape[0]
    return float(total_loss / total_tokens)

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
        # Forward + backward
        logits, _, efe = model.process(obs=Xb, lab=Yb, adapt_synapses=True)
        loss = compute_loss(logits, Yb)
        total_loss += loss
        n_batches += 1

        if VERBOSITY >= 1 and n_batches % 10 == 0:
            print(f"Epoch {epoch} | Batch {n_batches} | Loss {float(loss):.4f}")

    avg_train_loss = total_loss / max(1, n_batches)
    val_loss = evaluate(model, val_loader)
    print(f"Epoch {epoch} done | Train Loss={float(avg_train_loss):.4f} | Val Loss={float(val_loss):.4f}")

    if (epoch + 1) % SAVE_POINT == 0:
        model.save_to_disk(params_only=True)

sim_time = time.time() - sim_start_time
print(f"Training completed in {sim_time/60:.2f} min.")

