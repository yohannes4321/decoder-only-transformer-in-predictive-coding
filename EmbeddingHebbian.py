# from jax import numpy as jnp, random, jit
# from ngclearn.utils import tensorstats
# from ngcsimlib.compartment import Compartment
# from ngcsimlib.compilers.process import transition
# from ngclearn.components.jaxComponent import JaxComponent
# from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
# from flax import linen as nn
# from flax.core import freeze, unfreeze
# from flax import nnx
# import ngclearn.utils.weight_distribution as dist

# class EmbeddingHebbain(JaxComponent):
#     """
#     Embedding cell compatible with ngcsimlib:
#     - Computes token + positional embeddings
#     - Stateless (no rate dynamics)
#     - Compatible with @transition decorator
#     """

#     def __init__(self, name, dkey, n_embed,vocab_size,block_size,dropout_rate,batch_size,eta,wlb,wub,optim_type,**kwargs):
#         super().__init__(name, **kwargs)
#         self.name = name
#         self.batch_size = batch_size
#         self.block_size = block_size
#         self.n_embed = n_embed
#         self.flat_dim = batch_size * block_size
#         self.vocab_size = vocab_size
#         dkey, *subkeys = random.split(dkey, 10)
#         rngs = nnx.Rngs(default=dkey)
#         # token embedding
#         self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
#         # positional embedding
#         self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)
#         # dropout
#         self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

#         # compartments
#         self.j = Compartment(jnp.zeros((batch_size, block_size), dtype=jnp.int32))  # token indices input
#         self.emb_out = Compartment(jnp.zeros((batch_size, block_size, n_embed))) 
#         self.W_emb_q=HebbianSynapse(
#          "W_emb_q", shape=(self.n_embed,self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
#                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
#         )
        

#         self.W_emb_k=HebbianSynapse(
#         "W_emb_k", shape=(self.n_embed, self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
#         bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
#          )
#         self.W_emb_v=HebbianSynapse(
#         "W_emb_v", shape=(self.n_embed, self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
#         bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
#         ) # output embeddings
#     def embedingfuncion(j):
#         B, T = self.j.value.shape
#         # token embeddings
#         tok_emb = self.wte(self.j.value)  # (B, T, n_embed)
#         # positional embeddings
#         pos = jnp.arange(T)[None, :]
#         pos_emb = self.wpe(pos)  # (1, T, n_embed)
#         # sum token + positional embeddings
#         drive = tok_emb + pos_emb
#         # apply dropout
#         drive = self.dropout(drive)
#         self.emb_out.set(drive)
#         return self.emb_out
    
#     @transition(output_compartments=["emb_out"])
#     @staticmethod
#     def advance_state(self, dt=1.):
#         """Compute token + positional embeddings"""
#         return embedingfuncion(j)

        
            

#     @transition(output_compartments=["emb_out", "j"])
#     @staticmethod
#     def reset(self):
#         """Reset input tokens and embeddings to zeros"""
#         j_val=self.j.set(jnp.zeros((self.batch_size, self.block_size), dtype=jnp.int32))
#         emb_reset_out=self.emb_out.set(jnp.zeros((self.batch_size, self.block_size, self.n_embed)))
#         return j_val,emb_out


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
    EmbeddingSynapse component:
    - Creates token and positional embeddings
    - Produces (B, S, D) embeddings and flattens to (B*S, D)
    - Includes Hebbian learning weight update
    """

    def __init__(self, name, dkey, vocab_size, n_embed, block_size,
                 batch_size, resist_scale=1., weight_init=None, shape=None,
                 eta=0., w_decay=0., w_bound=1., weights=None, **kwargs):
        super().__init__(name, **kwargs)

        # Store hyperparameters
        self.name = name
        # self.dt = dt
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

        # Create token and positional embeddings
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)

        # Setup compartments
        self.inputs = Compartment(None)
        self.outputs = Compartment(None)
        self.pre = Compartment(None)
        self.post = Compartment(None)
        self.weights = Compartment(None)
        self.eta = Compartment(jnp.ones((1, 1)) * eta)

       

    def _init(self, weights):
        self.rows = weights.shape[0]
        self.cols = weights.shape[1]
        preVals = jnp.zeros((self.batch_size, self.rows))
        postVals = jnp.zeros((self.batch_size, self.cols))
        self.inputs.set(preVals)
        self.outputs.set(postVals)
        self.pre.set(preVals)
        self.post.set(postVals)
        self.weights.set(weights)

    # --- Core computation ---
    @transition(output_compartments=["outputs"])
    @staticmethod
    def advance_state(Rscale, w_bounds, w_decay, inputs, weights,
                       pre, post, eta, wte, wpe, block_size):
        """
        Combines token and positional embeddings, flattens (B, S, D) -> (B*S, D)
        and applies optional Hebbian weight update.
        """
        # Assume inputs are integer token IDs (B, S)
        batch_size, seq_len = inputs.shape

        # Token embeddings (B, S, D)
        token_embeds = wte[inputs]

        # Positional embeddings (S, D)
        pos_embeds = wpe[jnp.arange(seq_len)]

        # Combine embeddings
        embeddings = token_embeds + pos_embeds  # (B, S, D)

        # Flatten to (B*S, D)
        outputs = embeddings.reshape(batch_size * seq_len, -1)

        # --- Hebbian weight update ---
        

        return outputs

    @transition(output_compartments=["inputs", "outputs", "pre", "post", "eta"])
    @staticmethod
    def reset(batch_size, rows, cols, eta0):
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
        file_name = f"{directory}/{self.name}.npz"
        jnp.savez(file_name, weights=self.weights.value)

    def load(self, directory, **kwargs):
        file_name = f"{directory}/{self.name}.npz"
        data = jnp.load(file_name)
        self._init(data['weights'])

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
