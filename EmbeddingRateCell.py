from jax import numpy as jnp, random, jit
from ngclearn.utils import tensorstats
from ngcsimlib.compartment import Compartment
from ngcsimlib.compilers.process import transition
from ngclearn.components.jaxComponent import JaxComponent
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
from flax import linen as nn
from flax.core import freeze, unfreeze
from flax import nnx
import ngclearn.utils.weight_distribution as dist

class Embedding(JaxComponent):
    """
    Embedding cell compatible with ngcsimlib:
    - Computes token + positional embeddings
    - Stateless (no rate dynamics)
    - Compatible with @transition decorator
    """

    def __init__(self, name, dkey, n_embed,vocab_size,block_size,dropout,batch_size,eta,wlb,wub,optim_type,**kwargs):
        super().__init__(name, **kwargs)
        self.name = name
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = batch_size * block_size
        self.vocab_size = vocab_size
        dkey, *subkeys = random.split(dkey, 10)
        rngs = nnx.Rngs(default=dkey)
        # token embedding
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        # positional embedding
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)
        # dropout
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

        # compartments
        self.j = Compartment(jnp.zeros((batch_size, block_size), dtype=jnp.int32))  # token indices input
        self.emb_out = Compartment(jnp.zeros((batch_size, block_size, n_embed))) 
        self.W_emb_q=HebbianSynapse(
         "W_emb_q", shape=(self.n_embed,self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        )
        

        self.W_emb_k=HebbianSynapse(
        "W_emb_k", shape=(self.n_embed, self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
         )
        self.W_emb_v=HebbianSynapse(
        "W_emb_v", shape=(self.n_embed, self.n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        ) # output embeddings
    def embedingfuncion(j):
        B, T = self.j.value.shape
        # token embeddings
        tok_emb = self.wte(self.j.value)  # (B, T, n_embed)
        # positional embeddings
        pos = jnp.arange(T)[None, :]
        pos_emb = self.wpe(pos)  # (1, T, n_embed)
        # sum token + positional embeddings
        drive = tok_emb + pos_emb
        # apply dropout
        drive = self.dropout(drive)
        self.emb_out.set(drive)
        return self.emb_out
    
    @transition(output_compartments=["emb_out"])
    @staticmethod
    def advance_state(self, dt=1.):
        """Compute token + positional embeddings"""
        return embedingfuncion(j)

        
            

    @transition(output_compartments=["emb_out", "j"])
    @staticmethod
    def reset(self):
        """Reset input tokens and embeddings to zeros"""
        j_val=self.j.set(jnp.zeros((self.batch_size, self.block_size), dtype=jnp.int32))
        emb_reset_out=self.emb_out.set(jnp.zeros((self.batch_size, self.block_size, self.n_embed)))
        return j_val,emb_out