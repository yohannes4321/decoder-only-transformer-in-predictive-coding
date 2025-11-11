from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from ngclearn.components.jaxComponent import JaxComponent
from ngcsimlib.compartment import Compartment
from jax import numpy as jnp, random, jit
from ngcsimlib.compilers.process import transition
import jax
from functools import partial
from config import Config as config
import ngclearn.utils.weight_distribution as dist
@partial(jit, static_argnums=[4, 5, 6])
def _compute_attention(Q, K, V, mask, n_heads, d_head, dropout_rate, key):
    """
    Compute multi-head attention
    """
    if Q.ndim == 2:
        # 2D input: (batch_size * seq_len, n_embed) -> reshape to 3D
        M, D = Q.shape
        B = config.batch_size       
        S = M // B  # batch_size
        Q_3d = Q.reshape(B, S, D)
        K_3d = K.reshape(B, S, D)
        V_3d = V.reshape(B, S, D)
    else:
        # 3D input: (batch_size, seq_len, n_embed)
        B, S, D = Q.shape
        Q_3d, K_3d, V_3d = Q, K, V
    # Reshape for multi-head attention
    q = Q.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    k = K.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3]) 
    v = V.reshape((B, S, n_heads, d_head)).transpose([0, 2, 1, 3])
    
    # Scaled dot-product attention
    score = jnp.einsum("BHTE,BHSE->BHTS", q, k) / jnp.sqrt(d_head)
    
    if mask is not None:
        Tq, Tk = q.shape[2], k.shape[2]
        _mask = mask.reshape((B, 1, Tq, Tk))
        score = jnp.where(_mask, score, -jnp.inf)
        
    score = jax.nn.softmax(score, axis=-1)
    score = score.astype(q.dtype)
    
    if dropout_rate > 0.0:
        # dkey = random.fold_in(key, 0)
        dkey = random.PRNGKey(0)
        score = jax.random.bernoulli(dkey, 1 - dropout_rate, score.shape) * score / (1 - dropout_rate)
        
    attention = jnp.einsum("BHTS,BHSE->BHTE", score, v)
    attention = attention.transpose([0, 2, 1, 3]).reshape((B, S, -1))
    print(attention)
    return attention

class AttentionBlock(JaxComponent):
    """
    Multi-head attention block for NGC attention.
    
    Takes Q, K, V inputs and computes scaled dot-product attention 
    with optional masking and dropout.
    
    | --- Compartments: ---
    | inputs_q - query inputs
    | inputs_k - key inputs  
    | inputs_v - value inputs
    | mask - attention mask
    | outputs - attention outputs
    | key - JAX PRNG key

    Args:
        name: Component name
        n_heads: Number of attention heads
        n_embed: Embedding dimension
        seq_len: Sequence length
        dropout_rate: Attention dropout rate
        batch_size: Batch size
    """
    
    def __init__(self,name, dkey,n_heads, n_embed, seq_len, eta, wlb, wub, optim_type, act_fx,tau_m,dim, dropout_rate=0.5, batch_size=1, **kwargs):
        super().__init__(name, **kwargs)

        dkey, *subkeys = random.split(dkey, 10)
        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_head = n_embed // n_heads
        self.flat_dim = seq_len * n_embed
        # Input compartments
        self.inputs_q = RateCell("inputs_q",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        self.input_q_ErrorCell=ErrorCell("input_q_ErrorCell",n_units=dim)

        self.inputs_k = RateCell("inputs_k",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        self.input_k_ErrorCell=ErrorCell("input_k_ErrorCell",n_units=dim)
        self.inputs_v = RateCell("inputs_v",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        self.input_v_ErrorCell=ErrorCell("input_v_ErrorCell",n_units=dim)
        self.mask = Compartment(jnp.zeros((batch_size, seq_len, seq_len), dtype=bool))
        self.inputs_k_attentionout = HebbianSynapse(
                    "inputs_k_attentionout", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
        self.inputs_q_attentionout = HebbianSynapse(
                    "inputs_q_attentionout", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
        self.inputs_v_attentionout = HebbianSynapse(
                    "inputs_v_attentionout", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )

        #static 

        self.Eattentionout_input_k = StaticSynapse(
                    "Eattentionout_inputk", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )

        self.Eattentionout_input_q = StaticSynapse(
                    "Eattentionout_inputq", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )
        self.attentionout_input_v = StaticSynapse(
                    "attentionout_inputv", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )



        
        
        
        self.key = Compartment(random.PRNGKey(0))
        self.flat_outputs=Compartment(jnp.zeros((batch_size,self.flat_dim)))
        # Output compartment
        self.outputs = Compartment(jnp.zeros((batch_size, seq_len, dim)))
        self.Attention_out=RateCell("Attention_out",n_units=n_embed,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        self.Attentionout_Error = ErrorCell("Attentionout_Error",n_units=dim)
        self.attention_to_mlp = HebbianSynapse(
                    "attention_to_mlp", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
        self.Emlp1_to_attention = StaticSynapse(
                    "mlp_to_attention", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )

    @transition(output_compartments=["outputs"])
    @staticmethod
    def advance_state(inputs_q, inputs_k, inputs_v, mask, n_heads, d_head, dropout_rate, key):
        """
        Compute multi-head attention
        """
        attention = _compute_attention(
            inputs_q, inputs_k, inputs_v, mask, n_heads, d_head, dropout_rate, key
        )
        
        return attention

    @transition(output_compartments=["inputs_q", "inputs_k", "inputs_v", "mask", "outputs"])
    @staticmethod
    def reset(batch_size, seq_len, n_embed):
        """
        Reset compartments to zeros
        """
        zeros_3d = jnp.zeros((batch_size, seq_len, n_embed))
        mask = jnp.zeros((batch_size, seq_len, seq_len), dtype=bool)
        return zeros_3d, zeros_3d, zeros_3d, mask, zeros_3d

    @classmethod
    def help(cls):
        """Component help function"""
        properties = {
            "component_type": "AttentionBlock - multi-head self-attention mechanism"
        }
        compartment_props = {
            "inputs": 
                {"inputs_q": "Query inputs (batch_size, seq_len, n_embed)",
                 "inputs_k": "Key inputs (batch_size, seq_len, n_embed)", 
                 "inputs_v": "Value inputs (batch_size, seq_len, n_embed)",
                 "mask": "Attention mask (batch_size, seq_len, seq_len)"},
            "outputs":
                {"outputs": "Attention outputs (batch_size, seq_len, n_embed)"},
        }
        hyperparams = {
            "n_heads": "Number of attention heads",
            "n_embed": "Embedding dimension", 
            "seq_len": "Sequence length",
            "dropout_rate": "Attention dropout rate",
            "batch_size": "Batch size dimension"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = MultiHeadAttention(Q, K, V, mask)",
                "hyperparameters": hyperparams}
        return info