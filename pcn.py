from Dataloader import vocab_size
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
from ngclearn.utils.optim import adam
from jax import jit,random,numpy as jnp 
# from ngclearn.com.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from flax import nnx
from flax import linen as nn
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from functools import partial as bind
block_size = 128
n_embd = 64
drop_out=0.5
batch_size=32



class TokenAndPostionalEmbedding(nnx.Module):
    def __init__(self,vocab_size,n_embd,block_size,dropout,batch_size,key,**kwargs):
        self.block_size=block_size
        self.vocab_size = vocab_size
        self.n_embd=n_embd
        rngs=nnx.Rngs(default=key)
        self.token_embed=nnx.Embed(num_embedding=vocab_size,features=n_embd,rngs=rngs)
        self.pos_emb=nnx.Embed(num_embedding=block_size,features=n_embd,rngs=rngs)
      
    def __call__(self,x):
        maxlen=jnp.shape(x)[-1]
        x=self.token_embed(x)
        postions=jnp.arange(start=0,stop=maxlen,step=1)
        y=self.pos_emb(postions)
        return x+ y





class feedforward_1():
    def __init__(self,):
        dkey1, dkey2 = random.split(dkey, 2)
        with Context("feedforward_1") as feedforward_1:
          
            self.emlp_1=ErrorCell("emlp_1", n_units=hid1_dim)
            self.mlp_1=RateCell("mlp_1",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type='euler')
            self.Whid1=HebbianSynapse(
                    "Whid1", shape=(hid1, 4* hid1), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
            # self.layernorm_2=nn.LayerNorm(epsilon=1e-6, use_scale=True, use_bias=True)
            self.Whid0.inputs << self.mlp_0.zF
            self.emlp_1.mu << self.Whid0.outputs
            self.emlp_1.target << self.mlp_1.z
            advance_process=(JaxProcess(name="advance_process")
                            >> self.mlp_1.advance_state
                            >> self.Whid1.advance_state
            
            )
            reset_process = (JaxProcess(name="reset_process")
                             >> self.mlp_1.reset
                            >> self.Whid1.reset
            )
        def _dynamic(self,process):
            @Context.dynamicCommand
            def clamp_input(x):
                self.mlp_1.j.set(x)
                
class feedforward_2():
    def __init__(self,):
        dkey1, dkey2 = random.split(dkey, 2)
        with Context("feedforward_2") as feedforward_2:

            self.emlp_2=ErrorCell("emlp_2", n_units=hid1_dim)
            self.mlp_2=RateCell("mlp_2",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type='euler')
            self.Whid2=HebbianSynapse(
                    "Whid2", shape=(hid1*4 , hid1), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
            self.Whid1.inputs << self.mlp_1.zF
            self.emlp_2.mu << self.Whid1.outputs
            self.emlp_2.target << self.mlp_2.z

            advance_process=(JaxProcess(name="advance_process")
                            >> self.mlp_2.advance_state
                            >> self.Whid2.advance_state
            
            )
            reset_process = (JaxProcess(name="reset_process")
                             >> self.mlp_2.reset
                            >> self.Whid2.reset
            )
            def _dynamic(self,process):
                @Context.dynamicCommand
                def clamp_input(x):
                    self.mlp_2.j.set(x)
      


class selfAttention :
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
            self.attention.inputs << attentions

            advance_process=(JaxProcess(name="advance_process")
                            >> self.q.advance_state
                            >> self.k.advance_state
                            >> self.v.advance_state
            
            )
            reset_process = (JaxProcess(name="reset_process")
                             >> self.q.reset
                            >> self.k.reset
                            >> self.v.reset
            )

class PCN():
    def __init__(self, dkey, in_dim=1, out_dim=1, hid1_dim=128, hid2_dim=64, T=10,
                 dt=1., tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):



    

        hid1,hid1_dim,hid2,hid2_dim,hid3,hid4=hid1_dim
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 10)

        self.T = T
        self.dt = dt
        ## hard-coded meta-parameters for this model
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3
        wub = 0.3
        self.embedding=TokenAndPostionalEmbedding(vocab_size,n_embd,block_size,dropout,batch_size,dkey)
        self.feedforward_1=feedforward_1()
        self.feedforward_2=feedforward_2()
        self.layer_normalize=layer_normalize()
        self.attention=selfAttention(dkey,x1 ,mask ,n_heads=8,drop_out=0.5)

        if loadDir is not None:
            self.load_from_disk(loadDir)
        else :
            with Context("Circuit") as self.circuit:
                self.z_qkv=RateCell("z_qkv",n_units=in_dim,tau_m=0,act_fx="identity")

                self.z_score=RateCell("z_score",n_units=hid1,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")

                self.e_score=ErrorCell("e_score",n_units=hid1)
                self.z_fc1=RateCell("z_fc1",n_units=hid2 *4,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type="euler")
                self.e_fc1=ErrorCell("e_fc1",n_units=hid2)
                self.z_fc2=RateCell("z_fc2",n_units=hid3,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
                self.e_fc2=ErrorCell("e_fc2",n_units=hid3)
                self.zout=RateCell("zout",n_units=hid4,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
                self.e_zout=ErrorCell("e_zout",n_units=hid4)
                self.target_logits=RateCell("target_logits",n_units=out_dim,tau_m=0.,act_fx="identity")
                self.e_target_logits=ErrorCell("e_target_logits",n_units=out_dim)


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

                # assigning class 
                



                
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

                # --- RATE CELLS ---
                self.q_qkv = RateCell("q_qkv", n_units=in_dim, tau_m=0., act_fx="identity")
                self.q_score = RateCell("q_score", n_units=hid1, tau_m=0., act_fx=act_fx)
                self.q_fc1 = RateCell("q_fc1", n_units=hid2, tau_m=0., act_fx=act_fx)
                self.q_fc2 = RateCell("q_fc2", n_units=hid3, tau_m=0., act_fx=act_fx)
                self.q_out = RateCell("q_out", n_units=hid4, tau_m=0., act_fx=act_fx)
                self.qtarget_logits = RateCell("qtarget_logits", n_units=out_dim, tau_m=0., act_fx="identity")

                # --- ERROR CELL ---
                self.e_Qtarget = ErrorCell("e_Qtarget", n_units=out_dim)

                # --- STATIC SYNAPSES ---
                self.Qqkv_score = StaticSynapse(
                    "Qqkv_score", shape=(in_dim, hid1),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Qscore_fc1 = StaticSynapse(
                    "Qscore_fc1", shape=(hid1, hid2),
                    bias_init=dist.constant(value=0.), key=subkeys[1]
                )
                self.Qfc1_fc2 = StaticSynapse(
                    "Qfc1_fc2", shape=(hid2, hid3),
                    bias_init=dist.constant(value=0.), key=subkeys[2]
                )
                self.Qfc2_out = StaticSynapse(
                    "Qfc2_out", shape=(hid3, hid4),
                    bias_init=dist.constant(value=0.), key=subkeys[3]
                )
                self.Qout_target = StaticSynapse(
                    "Qout_target", shape=(hid4, out_dim),
                    bias_init=dist.constant(value=0.), key=subkeys[4]
                )

                # --- Wire the network ---
                self.Qqkv_score.inputs << self.q_qkv.zF
                self.q_score.j << self.Qqkv_score.outputs

                self.Qscore_fc1.inputs << self.q_score.zF
                self.q_fc1.j << self.Qscore_fc1.outputs

                self.Qfc1_fc2.inputs << self.q_fc1.zF
                self.q_fc2.j << self.Qfc1_fc2.outputs

                self.Qfc2_out.inputs << self.q_fc2.zF
                self.q_out.j << self.Qfc2_out.outputs

                self.Qout_target.inputs << self.q_out.zF
                self.qtarget_logits.j << self.Qout_target.outputs



                # wire self.Qout_target.outputs to 
                self.e_Qtarget.target << self.qtarget_logits.z
             

                advance_process = (
    JaxProcess(name="advance_process")
    >> self.Efc1_score.advance_state
    >> self.Efc2_fc1.advance_state
    >> self.Eout_fc2.advance_state
    >> self.Etarget_out.advance_state
    >> self.z_qkv.advance_state
    >> self.z_score.advance_state
    >> self.z_fc1.advance_state
    >> self.z_fc2.advance_state
    >> self.zout.advance_state
    >> self.target_logits.advance_state
    >> self.Wqkv_score.advance_state
    >> self.Wscore_fc1.advance_state
    >> self.Wfc1_fc2.advance_state
    >> self.Wfc2_zout.advance_state
    >> self.Wout_target.advance_state
    >> self.e_score.advance_state
    >> self.e_fc1.advance_state
    >> self.e_fc2.advance_state
    >> self.e_zout.advance_state
    >> self.e_target_logits.advance_state
)


                reset_process = (
    JaxProcess(name="reset_process")
    >> self.z_qkv.reset
    >> self.z_score.reset
    >> self.z_fc1.reset
    >> self.z_fc2.reset
    >> self.zout.reset
    >> self.target_logits.reset
    >> self.e_Qtarget.reset
    >> self.q_qkv.reset
    >> self.q_score.reset
    >> self.q_fc1.reset
    >> self.q_fc2.reset
    >> self.q_out.reset
    > self.qtarget_logits
    >> self.e_score.reset
    >> self.e_fc1.reset
    >> self.e_fc2.reset
    >> self.e_zout.reset
    >> self.e_target_logits.reset
)
                evolve_process = (
    JaxProcess(name="evolve_process")
    >> self.Wqkv_score.evolve
    >> self.Wscore_fc1.evolve
    >> self.Wfc1_fc2.evolve
    >> self.Wfc2_zout.evolve
    >> self.Wout_target.evolve
)
                project_process = (
    JaxProcess(name="project_process")
    >> self.q_qkv.advance_state
    >> self.Qqkv_score.advance_state
    >> self.q_score.advance_state
    >> self.Qscore_fc1.advance_state
    >> self.q_fc1.advance_state
    >> self.Qfc1_fc2.advance_state
    >> self.q_fc2.advance_state
    >> self.Qfc2_out.advance_state
    >> self.q_out.advance_state
    >> self.Qout_target.advance_state
    >> self.qtarget_logits.advance_state
    >> self.e_Qtarget.advance_state
)

                processes = (reset_process, advance_process, evolve_process, project_process)

                self._dynamic(processes)
    def _dynamic(self, processes):  # create dynamic commands for transformer circuit
        vars = self.circuit.get_components(
            # Rate cells
            "q_qkv", "q_score", "q_fc1", "q_fc2", "q_out", "qtarget_logits","e_Qtarget"

            # Error cells
            "e_score", "e_fc1", "e_fc2", "e_zout", "e_target_logits",

            # Projection StaticSynapses
            "Qqkv_score", "Qscore_fc1", "Qfc1_fc2", "Qfc2_out", "Qout_target",

            # Hebbian forward synapses
            "Wqkv_score", "Wscore_fc1", "Wfc1_fc2", "Wfc2_zout", "Wout_target",

            # Error synapse adapters
            "Efc1_score", "Efc2_fc1", "Eout_fc2", "Etarget_out",

            # Main dynamical z-states
            "z_qkv", "z_score", "z_fc1", "z_fc2", "zout", "target_logits"
        )
        (self.q_qkv, self.q_score, self.q_fc1, self.q_fc2, self.q_out, self.qtarget_logits, self.e_Qtarget,
        self.e_score, self.e_fc1, self.e_fc2, self.e_zout, self.e_target_logits,
        self.Qqkv_score, self.Qscore_fc1, self.Qfc1_fc2, self.Qfc2_out, self.Qout_target,
        self.Wqkv_score, self.Wscore_fc1, self.Wfc1_fc2, self.Wfc2_zout, self.Wout_target,
        self.Efc1_score, self.Efc2_fc1, self.Eout_fc2, self.Etarget_out,
        self.z_qkv, self.z_score, self.z_fc1, self.z_fc2, self.zout, self.target_logits) = vars

        self.nodes = vars
        

        self.nodes = vars

        reset_proc, advance_proc, evolve_proc, project_proc = processes

        # Add commands to the circuit
        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")


    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            self.Wqkv_score.save(model_dir)
            self.Wscore_fc1.save(model_dir)
            self.Wfc1_fc2.save(model_dir)
            self.Wfc2_zout.save(model_dir)
            self.Wout_target.save(model_dir)

        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            #self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory):
        """
        Loads parameter/config values from disk to this model

        Args:
            model_directory: directory/path to saved model parameter/config values
        """
        print(" > Loading model from ",model_directory)
        with Context("Circuit") as self.circuit:
            self.circuit.load_from_dir(model_directory)
            processes = (
                self.circuit.reset_process, self.circuit.advance_process,
                self.circuit.evolve_process, self.circuit.project_process
            )
            self._dynamic(processes)

    def process(self,obs,lab,adapt_synapses=True):
        # projection,expectation,maximization
        eps=0.001
        _lab=jnp.clip(lab,eps,1-eps)
        self.circuit.reset()
        self.Qqkv_score.weights.set(self.Wqkv_score.weights.value)
        self.Qqkv_score.biases.set(self.Wqkv_score.biases.value)

        self.Qscore_fc1.weights.set(self.Wscore_fc1.weights.value)
        self.Qscore_fc1.biases.set(self.Wscore_fc1.biases.value)

        self.Qfc1_fc2.weights.set(self.Wfc1_fc2.weights.value)
        self.Qfc1_fc2.biases.set(self.Wfc1_fc2.biases.value)

        self.Qfc2_out.weights.set(self.Wfc2_zout.weights.value)
        self.Qfc2_out.biases.set(self.Wfc2_zout.biases.value)

        self.Qout_target.weights.set(self.Wout_target.weights.value)
        self.Qout_target.biases.set(self.Wout_target.biases.value)

        # feedback synapis 
        # --- pin/tie feedback/error synapses to transpose of forward Hebbian synapses ---
        self.Efc1_score.weights.set(jnp.transpose(self.Wscore_fc1.weights.value))
        self.Efc2_fc1.weights.set(jnp.transpose(self.Wfc1_fc2.weights.value))
        self.Eout_fc2.weights.set(jnp.transpose(self.Wfc2_zout.weights.value))
        self.Etarget_out.weights.set(jnp.transpose(self.Wout_target.weights.value))

        # perfom P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        self.circuit.project(t=0.,dt=1.)  # do projection/inference

        #initialize dynamics of generative model latents to projected states
        self.z_score.z.set(self.q_score)
        self.z_fc1.z.set(self.q_fc1)
        self.z_fc2.z.set(self.q_fc2)
        self.zout.z.set(self.q_out)

        # Note error escore=0 efc1=0 efc2=0 ezout=0 at initial states
        self.e_target_logits.dmu.set(self.e_Qtarget.dmu.value)
        self.e_target_logits.dtarget.set(self.e_Qtarget.dtarget.value)
        y_mu_inf=self.qtarget_logits.z

        EFE=0
        y_mu=0
        if adapt_synapses:
            for ts in range(0,self.T):
                self.circuit.clamp_input(obs)
                self.circuit.clamp_target(_lab)
                self.circuit.advance(t=ts,dt=1.)

            y_mu=self.e_target_logits.mu.value
            L1=self.e_score.L.value
            L2=self.e_fc1.L.value
            L3=self.e_fc2.L.value
            L4=self.e_zout.L.value
            L5=self.e_target_logits.L.value
            EFE=L1+L2+L3+L4+L5

            if adapt_synapses==True:
                self.circuit.evolve(t=self.T,dt=1.)
        return y_mu_inf,y_mu,EFE
        

        
