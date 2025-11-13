# HebbainEmbedding.py
from ngclearn import numpy as jnp
from ngcsimlib.logger import info, warn
from ngcsimlib.compilers.process import transition
from ngcsimlib.component import Component
from ngcsimlib.compartment import Compartment
from ngclearn.utils.weight_distribution import initialize_params
from ngclearn.utils import tensorstats
from jax import random
import flax.nnx as nnx

class EmbeddingHebbain(Component):
    """
    EmbeddingHebbain component:
    - Creates token and positional embeddings
    - Produces (B, S, D) embeddings and flattens to (B*S, D)
    - Builds a full weight matrix (D x D) and bias vector (D,) from embeddings
      (W is the averaged outer product / covariance-like matrix; b is mean embedding)
    - Exposes outputs, weights and biases as Compartments
    - Provides reset/save/load support
    """

    def __init__(self, name, dkey, vocab_size, n_embed, block_size,
                 batch_size, resist_scale=1., weight_init=None, shape=None,
                 eta=0., w_decay=0., w_bound=1., weights=None, **kwargs):
        super().__init__(name, **kwargs)

        # Store hyperparameters
        self.name = name
        self.Rscale = resist_scale
        self.weight_init = weight_init
        self.shape = shape
        self.w_bounds = w_bound
        self.w_decay = w_decay
        self.eta0 = eta
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.vocab_size = vocab_size

        # Initialize random keys
        dkey, *subkeys = random.split(dkey, 4)
        rngs = nnx.Rngs(default=subkeys[0])

        # Create token and positional embeddings (1D embedding vectors per token/position)
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)  # token embedding table
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)  # positional embedding table

        # Setup compartments
        self.inputs = Compartment(None)   # expects (B, S) token ids
        self.outputs = Compartment(None)  # will hold flattened embeddings (B*S, D)
        self.pre = Compartment(None)
        self.post = Compartment(None)
        # For compatibility with other components that expect a 'weights' compartment
        # we'll store the full projection matrix W (D x D) here.
        self.weights = Compartment(None)  # (D, D)
        # Bias vector derived from embeddings
        self.biases = Compartment(None)   # (D,)
        self.eta = Compartment(jnp.ones((1, 1)) * eta)

        # If initial weights provided, initialize compartments from them (optional)
        if weights is not None:
            self._init(weights)

    def _init(self, weights):
        """
        Initialize internal compartments given a weights matrix. This keeps backward
        compatibility with earlier API where a weight matrix could be passed in.
        We set rows/cols consistent with a 2D weight matrix.
        """
        # Expect weights to be a 2D array (rows x cols)
        assert weights.ndim == 2, "weights must be 2D"
        self.rows = weights.shape[0]
        self.cols = weights.shape[1]
        preVals = jnp.zeros((self.batch_size, self.rows))
        postVals = jnp.zeros((self.batch_size, self.cols))
        self.inputs.set(preVals)   # placeholder shape until real run
        self.outputs.set(postVals)
        self.pre.set(preVals)
        self.post.set(postVals)
        # set the weight matrix compartment (D x D)
        self.weights.set(weights)
        # biases init
        # self.biases.set(jnp.zeros((self.cols,)))

    # --- Core computation ---
    @transition(output_compartments=["outputs", "weights"])
    @staticmethod
    def advance_state(Rscale, w_bounds, w_decay, inputs, weights, pre, post, eta, wte, wpe, block_size):
        """
        Combines token and positional embeddings, flattens (B, S, D) -> (B*S, D),
        and derives a full weight matrix and bias vector from the embeddings.

        Returns:
            outputs: (B*S, D) flattened embeddings
            weights: (D, D) projection matrix computed from embeddings
            biases: (D,) mean embedding (acts as bias)
        """
        # inputs expected as integer token IDs (B, S)
        # defensive checks
        if inputs is None:
            raise ValueError("EmbeddingHebbain.advance_state got `inputs=None`")
        # inputs shape
        batch_size, seq_len = inputs.shape

        # Token embeddings (B, S, D)
        # flax.nnx.Embed returns an object that supports indexing like wte[token_ids]
        token_embeds = wte[inputs]            # (B, S, D)

        # Positional embeddings (S, D) then broadcast to (B, S, D)
        pos_idxs = jnp.arange(seq_len)
        pos_embeds = wpe[pos_idxs]            # (S, D)
        # Broadcast and add
        embeddings = token_embeds + pos_embeds  # (B, S, D)

        # Flatten to (B*S, D) to feed 2D-only downstream components
        embeddings_flat = embeddings.reshape(batch_size * seq_len, -1)  # (B*S, D)
        outputs = embeddings_flat  # (B*S, D)

        # --- Build full weight matrix W and bias vector b from embeddings ---
        # W := (embeddings_flat^T @ embeddings_flat) / (B*S)  -> (D, D)
        # b := mean over embeddings_flat -> (D,)
        n_samples = embeddings_flat.shape[0] * 1.0
        # avoid division by zero
        if n_samples == 0:
            raise ValueError("No tokens present to build weights/biases (B*S == 0).")
        # compute mean and covariance-ish outer product
        mean_emb = jnp.mean(embeddings_flat, axis=0)                    # (D,)
        # center embeddings to get covariance-like matrix
        centered = embeddings_flat - mean_emb                           # (B*S, D)
        W = (centered.T @ centered) / n_samples                         # (D, D)
        # Optionally clip/scale W to some bounds (if using w_bounds), here we apply a soft clip
        if w_bounds is not None and w_bounds > 0.0:
            W = jnp.clip(W, -w_bounds, w_bounds)

        # Bias vector is the mean embedding
        b = mean_emb  # (D,)

        # Return outputs and expose W and b via compartments
        return outputs, W

    @transition(output_compartments=["inputs", "outputs", "pre", "post", "eta"])
    @staticmethod
    def reset(batch_size, rows, cols, eta0):
        # For compatibility with other components we keep the earlier reset signature.
        # However, embeddings produce outputs shaped (batch_size*seq_len, D) at run-time,
        # so these zero shapes merely serve as placeholders for simulation initialization.
        preVals = jnp.zeros((batch_size, rows))
        postVals = jnp.zeros((batch_size, cols))
        return (
            preVals,  # inputs
            postVals,  # outputs
            preVals,  # pre
            postVals,  # post
            jnp.ones((1,1)) * eta0
        )

    def save(self, directory, **kwargs):
        """
        Save weights and biases to npz for later reload.
        """
        file_name = f"{directory}/{self.name}.npz"
        # weights and biases may be None if not yet computed; guard them
        W = getattr(self.weights, "value", None)
        b = getattr(self.biases, "value", None)
        # convert None to zero arrays if needed
        if W is None:
            W = jnp.zeros((self.n_embed, self.n_embed))
        if b is None:
            b = jnp.zeros((self.n_embed,))
        jnp.savez(file_name, weights=W, biases=b)

    def load(self, directory, **kwargs):
        """
        Load weights and biases (W: D x D, b: D) and call _init to set shapes.
        """
        file_name = f"{directory}/{self.name}.npz"
        data = jnp.load(file_name)
        W = data['weights']
        b = data['biases']
        # initialize compartments consistent with W shape
        self._init(W)
        self.weights.set(W)
        self.biases.set(b)

    def __repr__(self):
        comps = [var for var in dir(self) if Compartment.is_compartment(getattr(self, var))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = ", ".join([f"{k}: {v}" for k, v in stats.items()])
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines
