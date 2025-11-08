import Dataloading import vocab_size,N_EMBD,block_size
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
from ngclearn.utils.optim import adam
from jax import jit,random,numpy as jnp 
from ngclearn.com.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from flax import nnx
from flax import linen as nn
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from functools import partial as bind
class TokenAndPostionalEmbedding(nnx.Module):
    def __init__(self,vocab_size,n_embd,block_size,dropout,batch_size,key,**kwargs):
        self.block_size=block_size
        self.vocab_size = vocab_size
        self.n_embd=n_embd
        rngs=nnx.Rngs(default=key)
        self.token_embed=nnx.Embed(num_embedding=vocab_size,features=n_embd,rngs=rngs)
        self.pos_emb=nnx.Embed(num_embedding=block_size,features=n_embd,rngs=rngs)
        
    def __call__(self,x.jax.Array):
        maxlen=jnp.shape(x)[-1]
        x=self.token_embed(x)
        postions=jnp.arange(start=0,stop=maxlen,step=1)
        y=self.pos_emb(postions)
        return x+y


class feedforward_1(JaxComponent):
    def __init__(self,):
        dkey1, dkey2 = random.split(dkey, 2)
        with Context("feedforward_1") as feedforward_1:
          
            self.emlp_1=ErrorCell("emlp_1", n_units=hid1_dim)
            self.mlp_1=RateCell("mlp_1",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior("gaussian",0.),integration_type='euler')
            self.Whid1=HebbianSynapse(
                    "Whid1", shape=(hid1, 4* hid1), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
            # self.layernorm_2=nn.LayerNorm(epsilon=1e-6, use_scale=True, use_bias=True)
            self.Whid0.inputs << self.mlp_0.zF
            self.emlp_1.mu << self.Whid0.outputs
            self.emlp_1.target << self.mlp_1.z

        def _dynamic(self,process):
            @Context.dynamicCommand
            def clamp_input(x):
                self.z0.j.set(x)
                
class feedforward_2(JaxComponent):
    def __init__(self,):
        dkey1, dkey2 = random.split(dkey, 2)
        with Context("feedforward_2") as feedforward_2:

            self.emlp_2=ErrorCell("emlp_2", n_units=hid1_dim)
            self.mlp_2=RateCell("mlp_2",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior("gaussian",0.),integration_type='euler')
            self.Whid2=HebbianSynapse(
                    "Whid2", shape=(hid1*4 , hid1), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
            self.Whid1.inputs << self.mlp_1.zF
            self.emlp_2.mu << self.Whid1.outputs
            self.emlp_2.target << self.mlp_2.z


            def _dynamic(self,process):
                @Context.dynamicCommand
                def clamp_input(x):
                    self.z0.j.set(x)
      


class selfAttension :
    @bind(jax.jit,static_argnums=[5,6])
    def masked_fill(self,x: jax.Array,mask: jax.Array,value=0) -> jax.Array:
            return jnp.where(mask,jnp.broadcast_to(value,x.shape),x)
    def __init__(self,dkey,x1: jax.Array,mask: jax.Array,n_heads: int=8,drop_out: float =0.5) -> jax.Array:
        """Args:
            dkey: JAX key to trigger any internal noise (drop-out)

            params (tuple): tuple of parameters

            x1 (jax.Array): query sequence. Shape: (B, T, Dq)

            x2 (jax.Array): key-value sequence. Shape: (B, S, Dkv)

            mask (jax.Array): mask tensor. Shape: (B, T, S)

            n_heads (int, optional): number of attention heads. Defaults to 8.

            dropout_rate (float, optional): dropout rate. Defaults to 0.0.

        Returns:
            jax.Array: output of self-attention
        """
        with Context("attention") as attention:

            B,T,Dq=x1.shape
            self.q=HebbianSynapse("query",shape=(vocab_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[0])
            self.k=HebbianSynapse("key",shape=(vocab_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[1])
            self.v=HebbianSynapse("value",shape=(vocab_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[2])
            self.attention=HebbianSynapse("attention",shape=(vocab_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[3])
            self.q.inputs << x
            self.k.inputs << x
            self.v.inputs << x
            embeding_shape = q.shape[-1]
            head_size= embeding_shape // n_heads
            q=q.reshape((B,T,n_heads,head_size)).transpose([0,2,1,3])
            k=k.reshape((B,T,n_heads,head_size)).transpose([0,2,1,3])
            v=v.reshpe((B,T,n_heads,head_size)).transpose([0,2,1,3])
            score=jnp.einsum("BHTE,BHSE ->BHTS",q,k) / jnp.sqrt(head_size)
            if mask is not None:
                Tq,Tk=q.shape[2],k.shape[2]
                assert mask.shape== ((B,Tq,Tk),(mask.shape,(B,Tq,Tk)))
                score=masked_fill(score,mask,value= - jnp.inf)
            score=jax.nn.softmax(score,axis=-1) #(B,H,T,S)
            score=score.astype(q.dtype)


            if dropout_rate >0:
                score ,_ = drop_out(dkey,score,rate=dropout_rate)
            attention = jnp.einsum("BHTS,BHTS->BHTE",score,v)
            attentions=attention.transpose([0,2,1,3]).reshape(B,T,-1) #(B,T,H,E) =>(B,T,D)
            return self.attention.inputs << attentions

class PCN(JaxComponent):
    def __init__(model_name="pcn",T=10,self,tau_m=10,out_dim=out_dim,act_fx="tanh",dkey,embed_value,dt=1.,loadDir=None 
    ,eta=0.001,exp_dir="exp",hid1,hid2,hid3,hid4,in_dim,,vocab_size,n_embd,block_size,drop_out,**kwargs):
        dkey,*subkeys=random.split(dkey,10)
        self.T=10
        self.dt=dt
        optim_type="adam"
        makedir(exp_dir)
        makedir(exp_dir + "/filters")
        wlb = -0.3
        wub = 0.3
        self.embedding=TokenAndPostionalEmbedding(vocab_size,n_embd,block_size,dropout,batch_size,dkey)
        

        if loadDir is not None:
            self.load_from_disk(loadDir)
        else :
            with Context("Circuit") as self.circuit:
                self.z_qkv=RateCell("z_qkv",n_units=in_dim,tau_m=0,act_fx="identity")
                self.z_score=RateCell("z_score",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior("gaussian",0.),integration_type='euler')
                self.e_score=ErrorCell("e_score",n_units=hid1)
                self.z_fc1=RateCell("z_fc1",n_units=hid2 *4,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type="euler")
                self.e_fc1=ErrorCell("e_fc1",n_units=hid2)
                self.z_fc2=RateCell("z_fc2",n_units=hid3,tau_m=tau_m,act_fx=act_fx,prior("gaussian",0.),integration_type="euler")
                self.e_fc2=ErrorCell("e_fc2",n_units=hid3)
                self.zout=RateCell("zout",n_units=hid4,tau_m=tau_m,act_fx=act_fx,prior("gaussian",0.),integration_type="euler")
                self.e_zout=ErrorCell("e_zout"n_units=hid4)
                self.target_logits=RateCell("target_logits",n_units=out_dim,tau_m=0.,act_fx="identity")
                self.e_target_logits=ErrorCell("e_target_logits")


                # connection
                self.Wqkv_score = HebbianSynapse(
                    "Wqkv_score", shape=(in_dim, hid1_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Wscore_fc1 = HebbianSynapse(
                    "Wscore_fc1", shape=(hid1, hid2), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Wfc1_fc2 = HebbianSynapse(
                    "Wfc1_fc2", shape=(hid2, hid3), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Wfc2_zout = HebbianSynapse(
                    "Wfc2_zout", shape=(hid3, hid4), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Wout_target = HebbianSynapse(
                    "Wout_target", shape=(hid4,out_dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )



                #feedback
                self.Efc1_score = StaticSynapse(
                    "Efc1_score", shape=(hid2, hid1), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )
                self.Efc2_fc1 = StaticSynapse(
                    "Efc2_fc1", shape=(hid3, hid2), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )
                self.Eout_fc2 = StaticSynapse(
                    "Eout_fc2", shape=(hid4, hid3), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )
                self.Etarget_out = StaticSynapse(
                    "Etarget_out", shape=(hid4, out_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )

                self.z_qkv.
                
                self.Wqkv_score.inputs << self.z_qkv.zF
                self.e_score.mu <<self.Wqkv_score.outputs
                self.e_score.target << self.z_score.z

                self.Wscore_fc1.inputs << self.z_score.zF
                self.e_fc1.mu << self.Wscore_fc1.outputs
                self.e_fc1.target << self.z_fc1.z

                self.Wfc1_fc2.inputs << self.z_fc1.zF
                self.e_fc2.mu << self.Wfc1_fc2.outputs
                self.e_fc2.target << self.z_fc2.z

                self.Wfc2_zout.inputs << self.z_fc2.zF
                self.e_zout.mu << self.Wfc2_zout.outputs
                self.e_zout.target << self.zout.z

                self.Wout_target.inputs << self.zout.zF
                self.e_target_logits.mu << self.Wout_target.outputs
                self.e_target_logits.target << self.target_logits.z

                #feedback
                self.Efc1_score.inputs << self.e_fc1.dmu
                self.z_score.j  << self.Efc1_score.outputs
                self.z_score.j_td << self.e_score.dtarget


                self.Efc2_fc1.inputs << self.e_fc2.dmu
                self.z_fc1.j << self.Efc2_fc1.outputs
                self.z_fc1.j_t << self.e_fc1.dtarget


                self.Eout_fc2.inputs <<  self.e_zout.dmu
                self.z_fc2.j << self.Eout_fc2.outputs
                self.z_fc2.j_td<< self.e_fc2.dtarget


                self.Etarget_out.inputs << self.e_target_logits.dmu
                self.z_out.j << self.Etarget_out.outputs
                self.z_out.j_td << self.e_zout.dtarget

                #setup 2 factor hebbian update

                self.Wqkv_score << self.z_qkv.zF
                self.Wqkv_score << self.e_score.dmu


                self.Wscore_fc1 << self.z_score.zF
                self.Wscore_fc1 << self.e_fc1.dmu

                self.Wfc1_fc2 << self.z_fc1.zF
                self.Wfc1_fc2 << self.e_fc2.dmu

                self.Wfc2_zout << self.z_fc2.zF
                self.Wfc2_zout  << self.e_zout.dmu

                self.Wout_target << self.zout.zF
                self.Wout_target << self.e_target_logits.dmu





                #:TODO MANY THINGS NOT CODED 


        def _dynamic(self,process):

            #TODO MANY THINGS TO ADD 
            @Context.dynamicCommand
            def clamp_input(x):
                self.z_qkv.j.set(x)
                
                #TODO NOT qo is not done 
            def clamp_target(y):
                self.target_logits.j.set(y)


