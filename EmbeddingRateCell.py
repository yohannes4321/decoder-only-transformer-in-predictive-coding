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
    def __init__(self, name, dkey, n_embed, vocab_size, block_size,
                 dropout, batch_size, eta, wlb, wub, optim_type, **kwargs):

        super().__init__(name, **kwargs)

        # ✅ store name
        self.name = name

        # ✅ split keys inside (correct!)
        dkey, *subkeys = random.split(dkey, 10)
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.flat_dim = batch_size * block_size 
        self.vocab_size = vocab_size
        self.shape = (batch_size, block_size, n_embed)
        rngs = nnx.Rngs(default=dkey)
        self.wte = nnx.Embed(vocab_size, n_embed, rngs=rngs)
        self.wpe = nnx.Embed(block_size, n_embed, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        
        self.emb_out = Compartment(jnp.zeros((self.flat_dim, self.n_embed)))
        self.j = Compartment(jnp.zeros((batch_size, block_size)))
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
        )
    @transition(output_compartments=["emb_out"])
    @staticmethod
    def advance_state(self, j,wte,wpe, t=0., dt=1., tau=10.):
        if self.j.value is not None:
            B, T = self.j.value.shape
            tok_emb = self.wte(self.j.value)
            pos = jnp.arange(T)[None, :]
            pos_emb = self.wpe(pos)
            drive = tok_emb + pos_emb
           
         
            self.z_flat.set(drive.reshape(B*T, self.n_embed))
            
          
        return self.z_flat
    @transition(output_compartments=["emb_out","j"])
    @staticmethod
    def reset(self):
        
        self.j.set(jnp.zeros((self.batch_size, self.block_size)))
        self.emb_out.set(jnp.zeros((self.batch_size, self.flat_dim)))