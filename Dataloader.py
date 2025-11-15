import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import tiktoken
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
BLOCK_SIZE = 128
N_EMBD = 512
MAX_EPOCHS = 5

# -----------------------------
# Directory setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
TOKENIZED_DIR = os.path.join(DATA_DIR, "tokenized_ptb")
os.makedirs(TOKENIZED_DIR, exist_ok=True)

# -----------------------------
# Tokenizer
# -----------------------------
class BPETokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.pad_token_id = 0
        self.vocab_size = self.tokenizer.n_vocab

    def tokenize_and_save(self, subset_name):
        subset_path = os.path.join(DATA_DIR, f"{subset_name}.csv")
        if not os.path.exists(subset_path):
            raise FileNotFoundError(f"{subset_name}.csv not found in {DATA_DIR}")

        output_path = os.path.join(TOKENIZED_DIR, f"{subset_name}_ids.pkl")
        if os.path.exists(output_path):
            print(f"Tokenized IDs already exist for {subset_name}, skipping.")
            return

        sep_id = self.tokenizer.eot_token
        with open(subset_path, "r", encoding="utf-8") as f:
            tokenized = [self.tokenizer.encode(line.strip()) + [sep_id] for line in f if line.strip()]

        with open(output_path, "wb") as f:
            pickle.dump(tokenized, f)

        print(f"Tokenized {subset_name} and saved to {output_path}")


# -----------------------------
# Dataset class
# -----------------------------
class PennTreeBankDataset:
    def __init__(self, tokenized_file, tokenizer_dir, block_size):
        self.block_size = block_size
        path = os.path.join(tokenizer_dir, tokenized_file)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Tokenized file not found: {path}")

        with open(path, "rb") as f:
            self.sequences = pickle.load(f)

        # Keep only sequences longer than 1
        self.sequences = [seq for seq in self.sequences if len(seq) > 1]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = jnp.array(seq[:-1][:self.block_size], dtype=jnp.int32)
        target_ids = jnp.array(seq[1:][:self.block_size], dtype=jnp.int32)
        return {"input_ids": input_ids, "target_ids": target_ids}


# -----------------------------
# Pad collate function
# -----------------------------
def pad_collate_fn(batch, pad_token_id=0):
    input_seqs = [item["input_ids"] for item in batch]
    target_seqs = [item["target_ids"] for item in batch]

    max_len = max([seq.shape[0] for seq in input_seqs])

    input_padded = jnp.stack([
        jnp.pad(seq, (0, max_len - seq.shape[0]), constant_values=pad_token_id)
        for seq in input_seqs
    ])
    target_padded = jnp.stack([
        jnp.pad(seq, (0, max_len - seq.shape[0]), constant_values=pad_token_id)
        for seq in target_seqs
    ])

    return {"input_ids": input_padded, "target_ids": target_padded}


# -----------------------------
# DataLoader generator
# -----------------------------
def create_dataloader(dataset, batch_size, shuffle=True, pad_token_id=0):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [dataset[i] for i in batch_idx]
        yield pad_collate_fn(batch, pad_token_id=pad_token_id)


# -----------------------------
# Example Embedding model
# -----------------------------


# -----------------------------
# Example usage
# -----------------------------
bpe = BPETokenizer()
# Uncomment if tokenized files not already created
bpe.tokenize_and_save("train")
bpe.tokenize_and_save("valid")
bpe.tokenize_and_save("test")
vocab_size=bpe.vocab_size

# Create datasets
train_dataset = PennTreeBankDataset("train_ids.pkl", TOKENIZED_DIR, BLOCK_SIZE)
val_dataset = PennTreeBankDataset("valid_ids.pkl", TOKENIZED_DIR, BLOCK_SIZE)
# Create dataloaders
train_loader = create_dataloader(train_dataset, BATCH_SIZE, shuffle=True, pad_token_id=bpe.pad_token_id)
val_loader = create_dataloader(val_dataset, BATCH_SIZE, shuffle=False, pad_token_id=bpe.pad_token_id)

print(train_loader)

