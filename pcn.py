from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
import ngclearn.utils.weight_distribution as dist
from ngclearn.operations import summation as summ
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
from jax import numpy as jnp,random,jit
import jax
import numpy as np 
from jax import jit,random,numpy as jnp,lax,nn
from functools import partial as bind

wlb=-0.3
wub=0.
optim_type = "adam"
# creating seeding keys (jax style)
dkey=random.PRNGKey(1234)
dkey,*subkeys=random.split(dkey,2)

class Embedding:
    def __init__(self,vocab_size,vocab_size,n_head,block_size,tau_m,n_embd,eta,gamma,**kwargs):
        self.n_head=n_head
        self.head_size=n_embd//self.n_head
        with Context("ngc") as ngc:
            
            self.Z_word=RateCell("Z_word",n_units=vocab_size,tau_m=0,act_fx=act_fx,prior=("gaussian",gamma),integration_type="euler")
            self.Z_pos=RateCell("Z_pos",n_units=block_size,tau_m=0,act_fx=act_fx,prior=("gaussian",gamma),integration_type="euler")

            self.W_word=HebbianSynapse("W_word",shape=(vocab_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[0])
            self.W_pos=HebbianSynapse("W_pos",shape=(block_size,n_embd),eta=eta,weight_init=dist.uniform(amin=wlb,amax=wub),
                                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[1])


            self.Z_word = RateCell(
                    "z1", n_units=n_embd, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.),
                    integration_type="euler"
                )
            self.Z_pos = RateCell(
                    "z1", n_units=n_embd, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.),
                    integration_type="euler"
                )

            self.e_word=ErrorCell("e_word", n_units=n_embd)

            self.e_pos=ErrorCell("e_pos", n_units=n_embd)
            self.E_pos = StaticSynapse(
                    "E_pos", shape=(hid2_dim, hid1_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                )
            self.E_embed = StaticSynapse(
                    "E_embed", shape=(out_dim, hid2_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                )
            self.E_word = StaticSynapse(
                    "E_word", shape=(hid2_dim, hid1_dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[6]
                )
            self.sum_two_inputs=RateCell("summation",n_units=block_size,tau_m=0,act_fx=act_fx,prior=("gaussian",gamma),integration_type="euler")
            #start connecting 
            #word embediing 
            self.W_word.inputs<< self.Z_word.zF
            self.e_word.mu << self.self.W_word.outputs
            self.e_word.target=self.Z_word.z
            # postion 

            self.W_pos.inputs << self.Z_pos.zF
            self.e_pos.mu << self.W_pos.outputs
            self.e_pos.target << self.Z_pos.z
            # mi combinded for 
            self.sum_two_inputs=summ(self.Z_word.zF,self.Zpos.zF)
            self.W_embed.inputs << self.sum_two_inputs
            self.e_embed.mu << self.W_embed.outputs
            self.e_embed.target << self.Z_embed.z



            self.E_word.inputs << self.e_word.dmu
            self.Z_word.j << self.E_word.outputs
            self.Z_word.j_td << self.e_word.dtarget

            ### wire epos to zpos via W2.T and e1 to z1 via d/dz1
            self.E_pos.inputs << self.e_pos.dmu
            self.Z_pos.j << self.E_pos.outputs
            self.Z_pos.j_td << self.e_pos.dtarget

            ## wire e3 to z2 via W3.T and e2 to z2 via d/dz2
            self.E_embed.inputs << self.e_embed.dmu
            self.sum_two_inputs.j << self.E_embed.outputs
            self.sum_two_inputs.j_td << self.e_embed.dtarget



            ## setup W_word for its 2-factor Hebbian update
            self.W_word.pre  << self.Z_word.zF
            self.W_word.post << self.e_word.dmu

            self.W_pos.pre << self.Z_pos.zF
            self.W_pos.post << self.e_pos.dmu

            self.W_embed.pre << self.sum_two_inputs.zF
            self.W.embed.post <<  self.e_embed.dmu
            self.Q1 = StaticSynapse(
                    "Q1", shape=(in_dim, hid1_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q2 = StaticSynapse(
                    "Q2", shape=(hid1_dim, hid2_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q3 = StaticSynapse(
                    "Q3", shape=(hid2_dim, out_dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                ## wire q0 -(Q1)-> q1, q1 -(Q2)-> q2, q2 -(Q3)-> q3
                self.Q1.inputs << self.q0.zF
                self.q1.j << self.Q1.outputs
                self.Q2.inputs << self.q1.zF
                self.q2.j << self.Q2.outputs
                self.Q3.inputs << self.q2.zF
                self.q3.j << self.Q3.outputs

                    ## ADVANCE process: forward activity flow
        advance_process = (
            JaxProcess(name="advance_process")
            >> self.E_word.advance_state
            >> self.E_pos.advance_state
            >> self.E_embed.advance_state
            >> self.Z_word.advance_state
            >> self.Z_pos.advance_state
            >> self.sum_two_inputs.advance_state
            >> self.W_word.advance_state
            >> self.W_pos.advance_state
            >> self.e_word.advance_state
            >> self.e_pos.advance_state
        )

        ## RESET process: clear neuronal states and error cells
        reset_process = (
            JaxProcess(name="reset_process")
            >> self.Z_word.reset
            >> self.Z_pos.reset
            >> self.sum_two_inputs.reset
            >> self.e_word.reset
            >> self.e_pos.reset
        )

        ## EVOLVE process: apply Hebbian plasticity updates (learning)
        evolve_process = (
            JaxProcess(name="evolve_process")
            >> self.W_word.evolve
            >> self.W_pos.evolve
        )

        ## PROJECT process: handles Q-paths (feedforward static mappings)
        project_process = (
            JaxProcess(name="project_process")
            >> self.Q1.advance_state
            >> self.Q2.advance_state
            >> self.Q3.advance_state
        )

        ## Collect all processes
        processes = (reset_process, advance_process, evolve_process, project_process)

                    











            # Embedding layer Complited 
            @Context.dynamicCommand
            def clamp(x):
                self.Z_word.j.set(x)
                self.Z_pos.j.set(x)

            

class PcnTransformer:
    #TODO
    def masked_fill(x: jax.Array,mask: jax.Array,value=0) -> jax.Array:
        return jnp.where(mask,jnp.broadcast_to(value,x.shape),x)


    @bind(jax.jit,static_argnums=[5,6])
    def self_attention(dkey,params: tuple,x1: jax.Array,x2: jax.Array,mask: jax.Array,n_heads: int=8,drop_out: float =0.0) -> jax.Array:
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

            B,T,Dq=x1.shape
            _,s,Dkv=x2.shape

            Wq,bq,Wk,bk,Wv,bv,Wout,bout=params
            # normal linear transformation 
            q=x1 @ Wq + bq
            k=x2 @ Wq + bk
            v= x2 @ Wv + bv

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
            attention=attention.transpose([0,2,1,3]).reshape(B,T,-1) #(B,T,H,E) =>(B,T,D)
            return attention @ Wout + bout # (B,T,Dq)
    @bind(jax.jit,static_argnums=[4, 5, 6, 7, 8])
    def run_attention_probe(
        dkey, params, encodings, mask, n_heads: int, dropout: float = 0.0, use_LN=False, use_LN_input=False,
        use_softmax=True):
        

        dkey1,key2 =random.split(dkey,2)
        learnable_query, Wq, bq, Wk, bk, Wv, bv, Wout, bout,\
        Wqs, bqs, Wks, bks, Wvs, bvs, Wouts, bouts, Wlnattn_mu,\
        Wlnattn_scale, Whid1, bhid1, Wln_mu1, Wln_scale1, Whid2,\
        bhid2, Wln_mu2, Wln_scale2, Whid3, bhid3, Wln_mu3, Wln_scale3,\
        Wy, by, ln_in_mu, ln_in_scale, ln_in_mu2, ln_in_scale2 = params
        self_attn_params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)

        if use_LN_input:
            learnable_query=layer_normalize(learnable_query,ln_in_mu,ln_in_scale)
            encodings=layer_normalize(encodeings,ln_in_mu2,ln_in_scale2)
        features=self_attention(dkey1, cross_attn_params, learnable_query, encodings, mask, n_heads, dropout)

        resdual_connections=features
        if use_LN:
            features=layer_normalize(features,Wln_mu1,Wln_scale1)
        


        #MLP

        skip=features
        if use_LN:
            features=layer_normalize(features,Wln_mu1,Wln_scale1)
        features=jnp.matmul((features),Whid1)+bhid1
        features=gelu(features)
        if use_LN:
            features=layer_normalize(features,Wln_mu2,Wln_scale2)
        features=features+skip
        out=jnp.matmul(features,Wy)+by
        if use_softmax:
            outs=jax.nn.softmax(out)
        return outs,features




        # TODO not finished 






class AttentiveProbe(Probe):


    # TODO UPDATEING WEIGHT BASED ON LOCAL ERROR IS NOT YER FORMED FOR ATTENTION MODEL
    def __init__(
        self,dkey,seq_length,n_embd,out_dim,num_heads=8,head_size,
        batch_size=1,,use_LN=True,use_softmax=True,Dropout=0.5,block_size
        eta=0.0002,eta_decay=0.0,min_eta=1e-5,**kwargs):

        super().__init__(dkey,batch_size,**kwargs)
        assert head_size % num_heads == 0, f"`attn_dim` must be divisible by `num_heads`. Got {attn_dim} and {num_heads}."
        self.dkey, *subkeys = random.split(self.dkey, 26)
        self.num_heads = num_heads
        self.batch_size=batch_size
        self.n_embd = n_embd
        self.head_size=head_size
        self.out_dim = out_dim
        self.use_softmax = use_softmax
        self.use_LN = use_LN
        self.use_LN_input = use_LN_input
        self.dropout = dropout
        self.block_size=block_size

        sigma = 0.02

        Wq = random.normal(subkeys[0], (input_dim, head_size)) * sigma
        bq = random.normal(subkeys[1], (1, attn_dim)) * sigma
        Wk = random.normal(subkeys[2], (input_dim, head_size)) * sigma
        bk = random.normal(subkeys[3], (1, attn_dim)) * sigma
        Wv = random.normal(subkeys[4], (input_dim, head_size)) * sigma
        bv = random.normal(subkeys[5], (1, attn_dim)) * sigma
        Wout = random.normal(subkeys[6], (head_size, head_size)) * sigma
        bout = random.normal(subkeys[7], (1, head_size)) * sigma
        self_attn_params = (Wq, bq, Wk, bk, Wv, bv, Wout, bout)
        ln_in_mu = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        ln_in_scale = jnp.ones((1, learnable_query_dim))  ## LN parameter (applied to output of attention)
        self_attn_scaling_params=(Wq, bq, Wk, bk, Wv, bv, Wout, bout)
        
        self.mask=np.zeros(self.batch_size,self.block_size,self.block_size)

        #MLP PARAMETERS
        Whid1 = random.normal(subkeys[16], (learnable_query_dim, learnable_query_dim)) * sigma
        bhid1 = random.normal(subkeys[17], (1, learnable_query_dim)) * sigma
        Wln_mu1 = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        Wln_scale1 = jnp.ones((1, learnable_query_dim)) ## LN parameter
        Whid2 = random.normal(subkeys[18], (learnable_query_dim, learnable_query_dim * 4)) * sigma
        bhid2 = random.normal(subkeys[19], (1, learnable_query_dim * 4)) * sigma
        Wln_mu2 = jnp.zeros((1, learnable_query_dim)) ## LN parameter
        Wln_scale2 = jnp.ones((1, learnable_query_dim)) ## LN parameter
        Whid3 = random.normal(subkeys[20], (learnable_query_dim * 4, learnable_query_dim)) * sigma
        bhid3 = random.normal(subkeys[21], (1, learnable_query_dim)) * sigma
        Wln_mu3 = jnp.zeros((1, learnable_query_dim * 4)) ## LN parameter
        Wln_scale3 = jnp.ones((1, learnable_query_dim * 4)) ## LN parameter
        Wy = random.normal(subkeys[22], (learnable_query_dim, out_dim)) * sigma
        by = random.normal(subkeys[23], (1, out_dim)) * sigma
        mlp_params = (ln_in_mu,ln_in_scale,Whid1, bhid1, Wln_mu1, Wln_scale1, Whid2, bhid2, Wln_mu2, Wln_scale2, Whid3, bhid3, Wln_mu3, Wln_scale3, Wy, by)

        self.optim_params = adam.adam_init(self.probe_params)
        self.eta = eta #0.001
        self.eta_decay = eta_decay
        self.min_eta = min_eta



        

