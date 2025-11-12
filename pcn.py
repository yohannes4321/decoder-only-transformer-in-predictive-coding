from Dataloader import vocab_size
from ngcsimlib.compilers import compile_command, wrap_command
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.io_utils import makedir
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
from EmbeddingHebbian import EmbeddingHebbain
from ngclearn.utils.optim import adam
from jax import jit,random,numpy as jnp 
import jax
from ngcsimlib.utils import get_current_context
from MLP import MLP
from attention_utils import AttentionBlock
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
# from ngclearn.com.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from flax import nnx
from flax import linen as nn
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from functools import partial as bind




class PCN():
    def __init__(self, dkey, block_size,n_embed,batch_size,n_heads,dropout_rate,dim,T=10,
                 dt=1., tau_m=10., act_fx="tanh", eta=0.001, exp_dir="exp",
                 model_name="pc_disc", loadDir=None, **kwargs):



       
        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        # dkey, *subkeys = random.split(dkey, 10)
       
        dkey, *subkeys = random.split(dkey, 10)
        self.T = T
        self.dt = dt
        ## hard-coded meta-parameters for this model
        optim_type = "adam"
        wlb = -0.3
        wub = 0.3
        
        

        
        if loadDir is not None:
            self.load_from_disk(loadDir)
        else :


            with Context("Circuit") as self.circuit:
                
         

                self.MLP = MLP("mlp", dkey, dim, act_fx, tau_m, eta, wlb, wub, optim_type)

                self.Attention=AttentionBlock("Attention",dkey,n_heads=n_heads,n_embed=n_embed,seq_len=14,dropout_rate=dropout_rate,batch_size=batch_size,tau_m=tau_m,dim=dim,eta=0.001,wlb=wlb,wub=wub,optim_type=optim_type,act_fx='tanh')
                self.Embedding = RateCell("Embedding", n_units=dim, tau_m=0., act_fx="identity")
                self.zqkv = RateCell("zqkv", n_units=dim, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.), integration_type="euler")
                self.zqkv_error = ErrorCell("zqkv_error", n_units=dim)

                self.EmbddingHebbain = EmbeddingHebbain("EmbddingHebbain", dkey, n_embed=n_embed,vocab_size=vocab_size,block_size=block_size,dropout_rate=dropout_rate,batch_size=batch_size,eta=eta,wlb=wlb,wub=wub,optim_type=optim_type,**kwargs )
                self.z_out_error=ErrorCell("z_out_error", n_units=dim)
                self.z_out=RateCell("z_out",n_units=dim,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type='euler')
                self.zout_targetlogit=HebbianSynapse(
                        "zout_targetlogit", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                        bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                    )
                self.Etargetlogit_zout = StaticSynapse(
                    "Etargetlogit_zout", shape=(dim, dim), bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.target_logit_error=ErrorCell("target_logit_error", n_units=dim)
                self.target_logit =RateCell("target_logit",n_units=dim,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type='euler')

                self.W_zqkv_q=HebbianSynapse(
                "W_zqkv_q", shape=(n_embed,n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                            bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                

                self.W_zqkv_k=HebbianSynapse(
                "W_zqkv_k", shape=(n_embed, n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.W_zqkv_v=HebbianSynapse(
                "W_zqkv_v", shape=(n_embed, n_embed), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )

                

                # forward connection
                self.EmbddingHebbain.inputs << self.Embedding.zF
                self.zqkv_error.mu<< self.EmbddingHebbain.outputs
                self.zqkv_error.target << self.zqkv.z

              





                self.W_zqkv_q.inputs << self.zqkv.zF
                # self.Attention.inputs_q.j << self.W_zqkv_q.outputs # z is needed to make attention compuation 
                self.Attention.input_q_ErrorCell.mu << self.W_zqkv_q.outputs
                self.Attention.input_q_ErrorCell.target << self.Attention.inputs_q.z

                self.W_zqkv_k.inputs << self.zqkv.zF
                # self.Attention.inputs_k.j << self.W_zqkv_k.outputs
                self.Attention.input_k_ErrorCell.mu << self.W_zqkv_k.outputs
                self.Attention.input_k_ErrorCell.target << self.Attention.inputs_k.z


                self.W_zqkv_v.inputs << self.zqkv.zF
                # self.Attention.inputs_v.j << self.W_zqkv_v.outputs
                self.Attention.input_v_ErrorCell.mu << self.W_zqkv_v.outputs
                self.Attention.input_v_ErrorCell.target << self.Attention.inputs_v.z

                # TODO I may be correct or overwrite it and only take inputs_v_attentionout 
                self.Attention.inputs_k_attentionout.inputs << self.Attention.inputs_k.zF
                self.Attention.Attentionout_Error.mu << self.Attention.inputs_k_attentionout.outputs
                self.Attention.Attentionout_Error.target << self.Attention.Attention_out.z

                
                self.Attention.inputs_q_attentionout.inputs << self.Attention.inputs_q.zF
                self.Attention.Attentionout_Error.mu << self.Attention.inputs_q_attentionout.outputs
                self.Attention.Attentionout_Error.target << self.Attention.Attention_out.z
                
                self.Attention.inputs_v_attentionout.inputs << self.Attention.inputs_v.zF
                self.Attention.Attentionout_Error.mu << self.Attention.inputs_v_attentionout.outputs
                self.Attention.Attentionout_Error.target << self.Attention.Attention_out.z
                #


                self.Attention.attention_to_mlp.inputs << self.Attention.Attention_out.zF
                self.MLP.mlp_1_error.mu << self.Attention.attention_to_mlp.outputs
                self.MLP.mlp_1_error.target << self.MLP.mlp_1.z


                self.MLP.mlp1_mlp2.inputs << self.MLP.mlp_1.zF
                self.MLP.mlp_2_error.mu << self.MLP.mlp1_mlp2.outputs
                self.MLP.mlp_2_error.target << self.MLP.mlp_2.z

                self.MLP.mlp2_zout.inputs << self.MLP.mlp_2.zF
                self.z_out_error.mu << self.MLP.mlp2_zout.outputs
                self.z_out_error.target << self.z_out.z 

                self.zout_targetlogit.inputs << self.z_out.zF
                self.target_logit_error.mu << self.zout_targetlogit.outputs
                self.target_logit_error.target << self.target_logit.z





                #TODO feedback start here 
                #feedback input_k,q,v and attention out
                self.Attention.Eattentionout_input_k.inputs << self.Attention.Attentionout_Error.dmu
                self.Attention.inputs_k.j  << self.Attention.Eattentionout_input_k.outputs
                self.Attention.inputs_k.j_td << self.Attention.input_k_ErrorCell.dtarget


                self.Attention.Eattentionout_input_q.inputs << self.Attention.Attentionout_Error.dmu
                self.Attention.inputs_q.j  << self.Attention.Eattentionout_input_q.outputs
                self.Attention.inputs_q.j_td << self.Attention.input_q_ErrorCell.dtarget

                self.Attention.Eattentionout_input_v.inputs << self.Attention.Attentionout_Error.dmu
                self.Attention.inputs_q.j  << self.Attention.Eattentionout_input_v.outputs
                self.Attention.inputs_q.j_td << self.Attention.input_v_ErrorCell.dtarget

                # feedback attention out  mlp 1
                self.MLP.Emlp1_to_attentionout.inputs << self.MLP.mlp_1_error.dmu
                self.Attention.Attention_out.j << self.MLP.Emlp1_to_attentionout.outputs
                self.Attention.Attention_out.j_td <<     self.Attention.Attentionout_Error.dtarget


                # feedback between mlp1 and mlp2
                self.MLP.Emlp2_mlp1.inputs << self.MLP.mlp_2_error.dmu
                self.MLP.mlp_1.j << self.MLP.Emlp2_mlp1.outputs
                self.MLP.mlp_1.j_td << self.MLP.mlp_1_error.dtarget

                # feedback between mlp2 and zout 
                self.MLP.Ezout_mlp2.inputs << self.z_out_error.dmu
                self.MLP.mlp_2.j << self.MLP.Ezout_mlp2.outputs
                self.MLP.mlp_2.j_td << self.MLP.mlp_2_error.dtarget
                # feedback between target and zout

                self.Etargetlogit_zout.inputs << self.target_logit_error.dmu
                self.z_out.j << self.Etargetlogit_zout.outputs
                self.z_out.j_td << self.z_out_error.dtarget

                #TODO 2 FACtor hebbian update
                self.W_zqkv_q.pre  << self.Embedding.zF
                self.W_zqkv_q.post << self.Attention.input_q_ErrorCell.dmu

                self.W_zqkv_k.pre << self.Embedding.zF
                self.W_zqkv_k.post << self.Attention.input_k_ErrorCell.dmu

                self.W_zqkv_v.pre << self.Embedding.zF
                self.W_zqkv_v.post << self.Attention.input_v_ErrorCell.dmu





                self.Attention.inputs_k_attentionout.pre << self.Attention.inputs_k.zF
                self.Attention.inputs_k_attentionout.post << self.Attention.Attentionout_Error.dmu  # if needed

                self.Attention.inputs_q_attentionout.pre << self.Attention.inputs_q.zF
                self.Attention.inputs_q_attentionout.post << self.Attention.Attentionout_Error.dmu  # if needed

                self.Attention.inputs_v_attentionout.pre << self.Attention.inputs_v.zF
                self.Attention.inputs_v_attentionout.post << self.Attention.Attentionout_Error.dmu  # if needed





                self.Attention.attention_to_mlp.pre << self.Attention.Attention_out.zF
                self.Attention.attention_to_mlp.post << self.MLP.mlp_1_error.dmu     # if needed

                
                self.MLP.mlp1_mlp2.pre << self.MLP.mlp_1.zF
                self.MLP.mlp1_mlp2.post  <<  self.MLP.mlp_2_error.dmu   

                self.MLP.mlp2_zout.pre    << self.MLP.mlp_2.zF
                self.MLP.mlp2_zout.post  << self.z_out_error.dmu

                self.zout_targetlogit.pre  << self.z_out.zF
                self.zout_targetlogit.post  << self.target_logit_error.dmu



                # finished 

                #TODO PROJECTION
                self.q_embed = RateCell("q_embed", n_units=dim, tau_m=0., act_fx="identity")
                self.q_zqkv = RateCell("q_zqkv", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_inputk = RateCell("q_inputk", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_inputv = RateCell("q_inputv", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_inputq = RateCell("q_inputq", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_Attentionscore = RateCell("q_Attentionscore", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_mlp1 = RateCell("q_mlp1", n_units=4 *dim, tau_m=0., act_fx=act_fx)
                self.q_mlp2 = RateCell("q_mlp2", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_out = RateCell("q_out", n_units=dim, tau_m=0., act_fx=act_fx)
                self.qtarget_logits = RateCell("qtarget_logits", n_units=dim, tau_m=0., act_fx="identity")
                self.qError_target_logit=ErrorCell("qError_target_logit", n_units=dim)
# #                 # --- ERROR CELL ---
#                 self.e_Qtarget = ErrorCell("e_Qtarget", n_units=dim)

              
                # --- STATIC SYNAPSES ---
                self.Qembed_zqkv = StaticSynapse(
                    "Qembed_zqkv", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q_zqkv_input_k = StaticSynapse(
                    "Q_zqkv_input_k", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q_zqkv_input_v = StaticSynapse(
                    "Q_zqkv_input_v", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Q_zqkv_input_q = StaticSynapse(
                    "Q_zqkv_input_q", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Qinputk_attentionscore = StaticSynapse(
                    "Qinputk_attentionscore", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Qinputq_attentionscore = StaticSynapse(
                    "Qinputq_attentionscore", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                self.Qinputv_attentionscore = StaticSynapse(
                    "Qinputv_attentionscore", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )


                self.QAttention_mlp1 = StaticSynapse(
                    "QAttention_mlp1", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[1]
                )

                # TODO I MAKE 4* 
                self.Qmlp1_mlp2= StaticSynapse(
                    "Qmlp1_mlp2", shape=(4* dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[2]
                )
                self.Qmlp2_out = StaticSynapse(
                    "Qmlp2_out", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[3]
                )
                self.Qout_target = StaticSynapse(
                    "Qout_target", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[4]
                )

                                # Embedding to K/V/Q projections
                self.Qembed_zqkv.inputs << self.q_embed.zF
                self.q_zqkv.j << self.Qembed_zqkv.outputs

                self.Q_zqkv_input_k.inputs << self.q_zqkv.zF
                self.q_inputk.j << self.Q_zqkv_input_k.outputs

                self.Q_zqkv_input_v.inputs << self.q_zqkv.zF
                self.q_inputv.j << self.Q_zqkv_input_v.outputs

                self.Q_zqkv_input_q.inputs << self.q_zqkv.zF
                self.q_inputq.j << self.Q_zqkv_input_q.outputs

                # Attention score receives projections from K, Q, V
                self.Qinputk_attentionscore.inputs << self.q_inputk.zF
                self.q_Attentionscore.j << self.Qinputk_attentionscore.outputs

                self.Qinputq_attentionscore.inputs << self.q_inputq.zF
                self.q_Attentionscore.j << self.Qinputq_attentionscore.outputs

                self.Qinputv_attentionscore.inputs << self.q_inputv.zF
                self.q_Attentionscore.j << self.Qinputv_attentionscore.outputs

                # Attention → MLP1
                self.QAttention_mlp1.inputs << self.q_Attentionscore.zF
                self.q_mlp1.j << self.QAttention_mlp1.outputs

                # MLP1 → MLP2
                self.Qmlp1_mlp2.inputs << self.q_mlp1.zF
                self.q_mlp2.j << self.Qmlp1_mlp2.outputs

                # MLP2 → OUT
                self.Qmlp2_out.inputs << self.q_mlp2.zF
                self.q_out.j << self.Qmlp2_out.outputs

                # OUT → target logits
                self.Qout_target.inputs << self.q_out.zF
                self.qtarget_logits.j << self.Qout_target.outputs
                #wire last error projection cell to target logit
                self.qError_target_logit.target << self.qtarget_logits.z

                advance_process = (JaxProcess(name="advance_process")
                        >> self.Embedding.advance_state
                        >> self.zqkv.advance_state
                        >> self.zqkv_error.advance_state
                        >> self.EmbddingHebbain.advance_state
                        >> self.Attention.Eattentionout_input_k.advance_state
                        >> self.Attention.Eattentionout_input_q.advance_state
                        >> self.Attention.Eattentionout_input_v.advance_state
                        >> self.MLP.Emlp1_to_attentionout.advance_state
                        >> self.MLP.Emlp2_mlp1.advance_state
                        >> self.MLP.Ezout_mlp2.advance_state
                        >> self.Etargetlogit_zout.advance_state
                        >> self.W_zqkv_q.advance_state 
                        >> self.W_zqkv_k.advance_state 
                        >> self.W_zqkv_v.advance_state
                        >> self.Embedding.advance_state
                        >> self.Attention.inputs_q.advance_state
                        >> self.Attention.input_q_ErrorCell.advance_state
                        >> self.Attention.inputs_k.advance_state
                        >> self.Attention.input_k_ErrorCell.advance_state
                        >> self.Attention.inputs_v.advance_state
                        >> self.Attention.input_v_ErrorCell.advance_state
                        >> self.Attention.inputs_k_attentionout.advance_state
                        >> self.Attention.inputs_q_attentionout.advance_state
                        >> self.Attention.inputs_v_attentionout.advance_state
                        >> self.Attention.Attentionout_Error.advance_state
                        >> self.Attention.attention_to_mlp.advance_state
                        >> self.MLP.mlp_1_error.advance_state
                        >> self.MLP.mlp_1.advance_state
                        >> self.MLP.mlp1_mlp2.advance_state
                        >> self.MLP.mlp_2_error.advance_state
                        >> self.MLP.mlp_2.advance_state
                        >> self.MLP.mlp2_zout.advance_state
                        >> self.z_out_error.advance_state
                        >> self.z_out.advance_state
                        >> self.zout_targetlogit.advance_state
                        >> self.target_logit_error.advance_state
                        >> self.target_logit.advance_state
                )





                reset_process = (JaxProcess(name="reset_process")

                            >> self.q_zqkv.reset
                            >> self.zqkv.reset
                            >> self.zqkv_error.reset
                            >> self.Embedding.reset
                            >> self.q_embed.reset
                            >> self.q_inputk.reset
                            >> self.q_inputv.reset
                            >> self.q_inputq.reset
                            >> self.q_Attentionscore.reset
                            >> self.q_mlp1.reset
                            >> self.q_mlp2.reset
                            >> self.q_out.reset
                            >> self.qtarget_logits.reset
                            >> self.qError_target_logit.reset
                            >> self.Attention.inputs_q.reset
                            >> self.Attention.input_q_ErrorCell.reset
                            >> self.Attention.inputs_k.reset
                            >> self.Attention.input_k_ErrorCell.reset
                            >> self.Attention.inputs_v.reset
                            >> self.Attention.input_v_ErrorCell.reset
                            >> self.Attention.Attentionout_Error.reset
                            >> self.MLP.mlp_1_error.reset
                            >> self.MLP.mlp_1.reset
                            >> self.MLP.mlp_2_error.reset
                            >> self.MLP.mlp_2.reset
                            >> self.z_out_error.reset
                            >> self.z_out.reset
                            >> self.target_logit_error.reset
                            >> self.target_logit.reset

                        )


                evolve_process = (
                        JaxProcess(name="evolve_process")
                        
                        >> self.W_zqkv_q.evolve
                        >> self.W_zqkv_k.evolve
                        >> self.W_zqkv_v.evolve

                        >> self.Attention.inputs_k_attentionout.evolve
                        >> self.Attention.inputs_q_attentionout.evolve
                        >> self.Attention.inputs_v_attentionout.evolve

                        >> self.Attention.attention_to_mlp.evolve

                        >> self.MLP.mlp1_mlp2.evolve
                        >> self.MLP.mlp2_zout.evolve

                        >> self.zout_targetlogit.evolve
                    )

                project_process = (
                    JaxProcess(name="project_process")

                    >> self.q_embed.advance_state
                    >> self.q_zqkv.advance_state
                    >> self.Qembed_zqkv.advance_state
                    # embed → K
                    >> self.Q_zqkv_input_k.advance_state
                    >> self.q_inputk.advance_state

                    # embed → V
                    >> self.Q_zqkv_input_v.advance_state
                    >> self.q_inputv.advance_state

                    # embed → Q
                    >> self.Q_zqkv_input_q.advance_state
                    >> self.q_inputq.advance_state

                    # K → AttentionScore
                    >> self.Qinputk_attentionscore.advance_state
                    # Q → AttentionScore
                    >> self.Qinputq_attentionscore.advance_state
                    # V → AttentionScore
                    >> self.Qinputv_attentionscore.advance_state

                    >> self.q_Attentionscore.advance_state

                    # AttentionScore → MLP1
                    >> self.QAttention_mlp1.advance_state
                    >> self.q_mlp1.advance_state

                    # MLP1 → MLP2
                    >> self.Qmlp1_mlp2.advance_state
                    >> self.q_mlp2.advance_state

                    # MLP2 → out
                    >> self.Qmlp2_out.advance_state
                    >> self.q_out.advance_state

                    # out → target logits
                    >> self.Qout_target.advance_state
                    >> self.qtarget_logits.advance_state

                    # final error cell
                    >> self.qError_target_logit.advance_state
                )
                process = (reset_process, advance_process, evolve_process, project_process)

                self._dynamic(process)
    def _dynamic(self,process):
        vars = self.circuit.get_components( 

        "Embedding",
        "zqkv",
        "zqkv_error",
        "EmbddingHebbain",
        "Eattentionout_input_k",
        "Eattentionout_input_q",
        "Eattentionout_input_v",
        "Emlp1_to_attentionout",
        "Emlp2_mlp1",
        "Ezout_mlp2",
        "Etargetlogit_zout",
        "W_zqkv_q",
        "W_zqkv_k",
        "W_zqkv_v",
        "inputs_q",
        "input_q_ErrorCell",
        "inputs_k",
        "input_k_ErrorCell",
        "inputs_v",
        "input_v_ErrorCell",
        "inputs_k_attentionout",
        "inputs_q_attentionout",
        "inputs_v_attentionout",
        "Attentionout_Error",
        "attention_to_mlp",
        "mlp_1_error",
        "mlp_1",
        "mlp1_mlp2",
        "mlp_2_error",
        "mlp_2",
        "mlp2_zout",
        "z_out_error",
        "z_out",
        "zout_targetlogit",
        "target_logit_error",
        "target_logit",
        "q_embed",
        "q_inputk",
        "q_inputv",
        "q_inputq",
        "q_Attentionscore",
        "q_mlp1",
        "q_mlp2",
        "q_out",
        "qtarget_logits",
        "qError_target_logit",
        "Q_zqkv_input_k",
        "Q_zqkv_input_v",
        "Q_zqkv_input_q",
        "Qinputk_attentionscore",
        "Qinputq_attentionscore",
        "Qinputv_attentionscore",
        "QAttention_mlp1",
        "Qmlp1_mlp2",
        "Qmlp2_out",
        "Qout_target"
        )

    # # Get components
    #     *component_names)

    # Unpack into attributes
        (
            self.Embedding,
            self.zqkv,
            self.zqkv_error,
            self.EmbddingHebbain,
            self.Eattentionout_input_k,
            self.Eattentionout_input_q,
            self.Eattentionout_input_v,
            self.Emlp1_to_attentionout,
            self.Emlp2_mlp1,
            self.Ezout_mlp2,
            self.Etargetlogit_zout,
            self.W_zqkv_q,
            self.W_zqkv_k,
            self.W_zqkv_v,
            self.inputs_q,
            self.input_q_ErrorCell,
            self.inputs_k,
            self.input_k_ErrorCell,
            self.inputs_v,
            self.input_v_ErrorCell,
            self.inputs_k_attentionout,
            self.inputs_q_attentionout,
            self.inputs_v_attentionout,
            self.Attentionout_Error,
            self.attention_to_mlp,
            self.mlp_1_error,
            self.mlp_1,
            self.mlp1_mlp2,
            self.mlp_2_error,
            self.mlp_2,
            self.mlp2_zout,
            self.z_out_error,
            self.z_out,
            self.zout_targetlogit,
            self.target_logit_error,
            self.target_logit,
            self.q_embed,
            self.q_inputk,
            self.q_inputv,
            self.q_inputq,
            self.q_Attentionscore,
            self.q_mlp1,
            self.q_mlp2,
            self.q_out,
            self.qtarget_logits,
            self.qError_target_logit,
            self.Q_zqkv_input_k,
            self.Q_zqkv_input_v,
            self.Q_zqkv_input_q,
            self.Qinputk_attentionscore,
            self.Qinputq_attentionscore,
            self.Qinputv_attentionscore,
            self.QAttention_mlp1,
            self.Qmlp1_mlp2,
            self.Qmlp2_out,
            self.Qout_target
        ) = vars

                # print(len(D))

        reset_proc, advance_proc, evolve_proc, project_proc = process

        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")
        @Context.dynamicCommand
        def clamp_input(x):
            self.Embedding.j.set(x)
            self.q_embed.j.set(x)
        # self.circuit.wrap_and_add_command((clamp_input), name="clamp_input")

        @Context.dynamicCommand
        def clamp_target(y):
            self.target_logit.j.set(y)
        # self.circuit.wrap_and_add_command((clamp_target), name="clamp_target")
        @Context.dynamicCommand
        def clamp_infer_target(y):
            self.qError_target_logit.target.set(y)
        # self.circuit.wrap_and_add_command((clamp_infer_target), name="clamp_infer_target")
        

    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter values to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
            # Assuming all these are attributes in your class
            self.EmbddingHebbain(model_dir)
            self.W_zqkv_q.save(model_dir)
            self.W_zqkv_k.save(model_dir)
            self.W_zqkv_v.save(model_dir)

            self.inputs_k_attentionout.save(model_dir)
            self.inputs_q_attentionout.save(model_dir)
            self.inputs_v_attentionout.save(model_dir)

            self.attention_to_mlp.save(model_dir)

            self.mlp1_mlp2.save(model_dir)
            self.mlp2_zout.save(model_dir)

            self.zout_targetlogit.save(model_dir)

        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
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

    def process(self, obs, lab, adapt_synapses=True):
        

        ## can think of the PCN as doing "PEM" -- projection, expectation, then maximization
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        #self.circuit.reset(do_reset=True)
        # self.circuit.reset()
        
            

        ## pin/tie inference synapses to be exactly equal to the forward ones
        # self.Qembed_zqkv.weights.set(self.EmbddingHebbain.weights.value)
        # self.Qembed_zqkv.biases.set(self.EmbddingHebbain.biases.value)
        self.Q_zqkv_input_q.weights.set(self.W_zqkv_q.weights.value)
        self.Q_zqkv_input_q.biases.set(self.W_zqkv_q.biases.value)

        self.Q_zqkv_input_k.weights.set(self.W_zqkv_k.weights.value)
        self.Q_zqkv_input_k.biases.set(self.W_zqkv_k.biases.value)

        self.Q_zqkv_input_v.weights.set(self.W_zqkv_v.weights.value)
        self.Q_zqkv_input_v.biases.set(self.W_zqkv_v.biases.value)

        self.Qinputk_attentionscore.weights.set(self.inputs_k_attentionout.weights.value)
        self.Qinputk_attentionscore.biases.set(self.inputs_k_attentionout.biases.value)

        self.Qinputq_attentionscore.weights.set(self.inputs_q_attentionout.weights.value)
        self.Qinputq_attentionscore.biases.set(self.inputs_q_attentionout.biases.value)

        self.Qinputv_attentionscore.weights.set(self.inputs_v_attentionout.weights.value)
        self.Qinputv_attentionscore.biases.set(self.inputs_v_attentionout.biases.value)

        self.QAttention_mlp1.weights.set(self.attention_to_mlp.weights.value)
        self.QAttention_mlp1.biases.set(self.attention_to_mlp.biases.value)

        self.Qmlp1_mlp2.weights.set(self.mlp1_mlp2.weights.value)
        self.Qmlp1_mlp2.biases.set(self.mlp1_mlp2.biases.value)

        self.Qmlp2_out.weights.set(self.mlp2_zout.weights.value)
        self.Qmlp2_out.biases.set(self.mlp2_zout.biases.value)

        self.Qout_target.weights.set(self.zout_targetlogit.weights.value)
        self.Qout_target.biases.set(self.zout_targetlogit.biases.value)

        ## pin/tie feedback synapses to transpose of forward ones
        self.Eattentionout_input_k.weights.set(jnp.transpose(self.inputs_k_attentionout.weights.value))
        self.Eattentionout_input_q.weights.set(jnp.transpose(self.inputs_q_attentionout.weights.value))
        self.Eattentionout_input_v.weights.set(jnp.transpose(self.inputs_v_attentionout.weights.value))

        self.Emlp1_to_attentionout.weights.set(jnp.transpose(self.attention_to_mlp.weights.value))
        self.Emlp2_mlp1.weights.set(jnp.transpose(self.mlp1_mlp2.weights.value))
        self.Ezout_mlp2.weights.set(jnp.transpose(self.mlp2_zout.weights.value))

        self.Etargetlogit_zout.weights.set(jnp.transpose(self.zout_targetlogit.weights.value))


        ## Perform P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        self.circuit.project(t=0., dt=1.) ## do projection/inference

        ## initialize dynamics of generative model latents to projected states
        self.inputs_k.z.set(self.q_inputk.z.value)
        self.inputs_v.z.set(self.q_inputv.z.value)
        self.inputs_q.z.set(self.q_inputq.z.value)
        self.Attention_out.z.set(self.q_Attentionscore.z.value)
        self.mlp_1.z.set(self.q_mlp1.z.value)
        self.mlp_2.z.set(self.q_mlp2.z.value)
        self.z_out.z.set(self.q_out.z.value)

        ## self.z3.z.set(self.q3.z.value)
        # ### Note: e1 = 0, e2 = 0 at initial conditions
        self.target_logit_error.dmu.set(self.qError_target_logit.dmu.value)
        self.target_logit_error.dtarget.set(self.qError_target_logit.dtarget.value)
        ## get projected prediction (from the P-step)
        y_mu_inf = self.qtarget_logits.z.value

        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses:
            ## Perform several E-steps
            for ts in range(0, self.T):
                self.circuit.clamp_input(obs) ## clamp data to z0 & q0 input compartments
                self.circuit.clamp_target(_lab) ## clamp data to e3.target
                self.circuit.advance(t=ts, dt=1.)

            y_mu = self.target_logit_error.mu.value ## get settled prediction
            ## calculate approximate EFE
            L1 = self.input_q_ErrorCell.L.value
            L2 = self.input_k_ErrorCell.L.value
            L3 = self.input_v_ErrorCell.L.value
            L4 = self.Attentionout_Error.L.value
            L5 = self.mlp_1_error.L.value
            L6 = self.mlp_2_error.L.value
            L7 = self.z_out_error.L.value
            L8 = self.target_logit_error.L.value

            EFE = L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8


            ## Perform (optional) M-step (scheduled synaptic updates)
            if adapt_synapses == True:
                #self.circuit.evolve(t=self.T, dt=self.dt)
                self.circuit.evolve(t=self.T, dt=1.)
        ## skip E/M steps if just doing test-time inference
        print(f"y_mu_inf -------------"  ,{y_mu_inf})
        print(f"y_mu -------------------- ",{y_mu})
        print(f"EFE ________________________",{EFE})
        return y_mu_inf, y_mu, EFE

    # def get_latents(self):
    #     return self.q2.z.value

    def _get_norm_string(self):  ## debugging routine
    # weights
        W_zqkv_q = self.W_zqkv_q.weights.value
        W_zqkv_k = self.W_zqkv_k.weights.value
        W_zqkv_v = self.W_zqkv_v.weights.value
        inputs_k_attentionout = self.inputs_k_attentionout.weights.value
        inputs_q_attentionout = self.inputs_q_attentionout.weights.value
        inputs_v_attentionout = self.inputs_v_attentionout.weights.value
        attention_to_mlp = self.attention_to_mlp.weights.value
        mlp1_mlp2 = self.mlp1_mlp2.weights.value
        mlp2_zout = self.mlp2_zout.weights.value
        zout_targetlogit = self.zout_targetlogit.weights.value

        # biases (if they exist)
        b_emb_q = self.W_zqkv_q.biases.value
        b_emb_k = self.W_zqkv_k.biases.value
        b_emb_v = self.W_zqkv_v.biases.value
        b_inputs_k_attentionout = self.inputs_k_attentionout.biases.value
        b_inputs_q_attentionout = self.inputs_q_attentionout.biases.value
        b_inputs_v_attentionout = self.inputs_v_attentionout.biases.value
        b_attention_to_mlp = self.attention_to_mlp.biases.value
        b_mlp1_mlp2 = self.mlp1_mlp2.biases.value
        b_mlp2_zout = self.mlp2_zout.biases.value
        b_zout_targetlogit = self.zout_targetlogit.biases.value

        _norms = (
            f"W_zqkv_q: {jnp.linalg.norm(W_zqkv_q)} "
            f"W_zqkv_k: {jnp.linalg.norm(W_zqkv_k)} "
            f"W_zqkv_v: {jnp.linalg.norm(W_zqkv_v)} "
            f"inputs_k_attentionout: {jnp.linalg.norm(inputs_k_attentionout)} "
            f"inputs_q_attentionout: {jnp.linalg.norm(inputs_q_attentionout)} "
            f"inputs_v_attentionout: {jnp.linalg.norm(inputs_v_attentionout)} "
            f"attention_to_mlp: {jnp.linalg.norm(attention_to_mlp)} "
            f"mlp1_mlp2: {jnp.linalg.norm(mlp1_mlp2)} "
            f"mlp2_zout: {jnp.linalg.norm(mlp2_zout)} "
            f"zout_targetlogit: {jnp.linalg.norm(zout_targetlogit)}\n"
            f"b_emb_q: {jnp.linalg.norm(b_emb_q)} "
            f"b_emb_k: {jnp.linalg.norm(b_emb_k)} "
            f"b_emb_v: {jnp.linalg.norm(b_emb_v)} "
            f"b_inputs_k_attentionout: {jnp.linalg.norm(b_inputs_k_attentionout)} "
            f"b_inputs_q_attentionout: {jnp.linalg.norm(b_inputs_q_attentionout)} "
            f"b_inputs_v_attentionout: {jnp.linalg.norm(b_inputs_v_attentionout)} "
            f"b_attention_to_mlp: {jnp.linalg.norm(b_attention_to_mlp)} "
            f"b_mlp1_mlp2: {jnp.linalg.norm(b_mlp1_mlp2)} "
            f"b_mlp2_zout: {jnp.linalg.norm(b_mlp2_zout)} "
            f"b_zout_targetlogit: {jnp.linalg.norm(b_zout_targetlogit)}"
        )
        
        return _norms

        







                                                    




                                                    


























                                    





                                    


                                    



                                    











                        

                        



        # #                 self.z_qkv=RateCell("z_qkv",n_units=dim,tau_m=0,act_fx="identity")

        #                 self.z_score=RateCell("z_score",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")

        #                 self.e_score=ErrorCell("e_score",n_units=dim)
        #                 self.z_fc1=RateCell("z_fc1",n_units=dim *4,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type="euler")
        #                 self.e_fc1=ErrorCell("e_fc1",n_units=dim)
        #                 self.z_fc2=RateCell("z_fc2",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        #                 self.e_fc2=ErrorCell("e_fc2",n_units=dim)
        #                 self.zout=RateCell("zout",n_units=dim,tau_m=tau_m,act_fx=act_fx,prior=("gaussian",0.),integration_type="euler")
        #                 self.e_zout=ErrorCell("e_zout",n_units=dim)
        #                 self.target_logits=RateCell("target_logits",n_units=dim,tau_m=0.,act_fx="identity")
        #                 self.e_target_logits=ErrorCell("e_target_logits",n_units=dim)


        #                 # connection
        #                 self.Wqkv_score = HebbianSynapse(
        #                     "Wqkv_score", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        #                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        #                 )
        #                 self.Wscore_fc1 = HebbianSynapse(
        #                     "Wscore_fc1", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        #                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        #                 )
        #                 self.Wfc1_fc2 = HebbianSynapse(
        #                     "Wfc1_fc2", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        #                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        #                 )
        #                 self.Wfc2_zout = HebbianSynapse(
        #                     "Wfc2_zout", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        #                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        #                 )
        #                 self.Wout_target = HebbianSynapse(
        #                     "Wout_target", shape=(dim,dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
        #                     bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
        #                 )



                        #feedback
                        # self.Efc1_score = StaticSynapse(
                        #     "Efc1_score", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                        # )
                        # self.Efc2_fc1 = StaticSynapse(
                        #     "Efc2_fc1", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                        # )
                        # self.Eout_fc2 = StaticSynapse(
                        #     "Eout_fc2", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                        # )
                    # self.Etarget_out = StaticSynapse(
                    #     "Etarget_out", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[4]
                    # )

    #                 # assigning class 
                    



                    
    #                 self.Wqkv_score.inputs << self.z_qkv.zF
    #                 self.e_score.mu <<self.Wqkv_score.outputs
    #                 self.e_score.target << self.z_score.z
                    
    #                 self.Wqkv_score.inputs << self.z_qkv.zF
    #                 self.e_score.mu <<self.Wqkv_score.outputs
    #                 self.e_score.target << self.z_score.z

    #                 self.Wqkv_score.inputs << self.z_qkv.zF
    #                 self.e_score.mu <<self.Wqkv_score.outputs
    #                 self.e_score.target << self.z_score.z


    #                 self.Wscore_fc1.inputs << self.z_score.zF
#                 self.e_fc1.mu << self.Wscore_fc1.outputs
#                 self.e_fc1.target << self.z_fc1.z

#                 self.Wfc1_fc2.inputs << self.z_fc1.zF
#                 self.e_fc2.mu << self.Wfc1_fc2.outputs
#                 self.e_fc2.target << self.z_fc2.z

#                 self.Wfc2_zout.inputs << self.z_fc2.zF
#                 self.e_zout.mu << self.Wfc2_zout.outputs
#                 self.e_zout.target << self.zout.z

#                 self.Wout_target.inputs << self.zout.zF
#                 self.e_target_logits.mu << self.Wout_target.outputs
#                 self.e_target_logits.target << self.target_logits.z

# #                 #feedback
# #                 self.Efc1_score.inputs << self.e_fc1.dmu
# #                 self.z_score.j  << self.Efc1_score.outputs
# #                 self.z_score.j_td << self.e_score.dtarget


# #                 self.Efc2_fc1.inputs << self.e_fc2.dmu
# #                 self.z_fc1.j << self.Efc2_fc1.outputs
# #                 self.z_fc1.j_t << self.e_fc1.dtarget


# #                 self.Eout_fc2.inputs <<  self.e_zout.dmu
# #                 self.z_fc2.j << self.Eout_fc2.outputs
# #                 self.z_fc2.j_td<< self.e_fc2.dtarget


# #                 self.Etarget_out.inputs << self.e_target_logits.dmu
# #                 self.z_out.j << self.Etarget_out.outputs
# #                 self.z_out.j_td << self.e_zout.dtarget

# #                 # #setup 2 factor hebbian update

# #                 # self.Wqkv_score << self.z_qkv.zF
# #                 # self.Wqkv_score << self.e_score.dmu


# #                 # self.Wscore_fc1 << self.z_score.zF
# #                 # self.Wscore_fc1 << self.e_fc1.dmu

# #                 # self.Wfc1_fc2 << self.z_fc1.zF
# #                 # self.Wfc1_fc2 << self.e_fc2.dmu

# #                 # self.Wfc2_zout << self.z_fc2.zF
# #                 # self.Wfc2_zout  << self.e_zout.dmu

# #                 # self.Wout_target << self.zout.zF
# #                 # self.Wout_target << self.e_target_logits.dmu

#                 # --- RATE CELLS ---
#                 self.q_qkv = RateCell("q_qkv", n_units=dim, tau_m=0., act_fx="identity")
#                 self.q_score = RateCell("q_score", n_units=dim, tau_m=0., act_fx=act_fx)
#                 self.q_fc1 = RateCell("q_fc1", n_units=dim, tau_m=0., act_fx=act_fx)
#                 self.q_fc2 = RateCell("q_fc2", n_units=dim, tau_m=0., act_fx=act_fx)
#                 self.q_out = RateCell("q_out", n_units=dim, tau_m=0., act_fx=act_fx)
#                 self.qtarget_logits = RateCell("qtarget_logits", n_units=dim, tau_m=0., act_fx="identity")

# # #                 # --- ERROR CELL ---
# # #                 self.e_Qtarget = ErrorCell("e_Qtarget", n_units=dim)

#                 # --- STATIC SYNAPSES ---
#                 self.Qqkv_score = StaticSynapse(
#                     "Qqkv_score", shape=(dim, dim),
#                     bias_init=dist.constant(value=0.), key=subkeys[0]
#                 )
#                 self.Qscore_fc1 = StaticSynapse(
#                     "Qscore_fc1", shape=(dim, dim),
#                     bias_init=dist.constant(value=0.), key=subkeys[1]
#                 )
#                 self.Qfc1_fc2 = StaticSynapse(
#                     "Qfc1_fc2", shape=(dim, dim),
#                     bias_init=dist.constant(value=0.), key=subkeys[2]
#                 )
#                 self.Qfc2_out = StaticSynapse(
#                     "Qfc2_out", shape=(dim, dim),
#                     bias_init=dist.constant(value=0.), key=subkeys[3]
#                 )
#                 self.Qout_target = StaticSynapse(
#                     "Qout_target", shape=(dim, dim),
#                     bias_init=dist.constant(value=0.), key=subkeys[4]
#                 )

# # #                 # --- Wire the network ---
# # #                 self.Qqkv_score.inputs << self.q_qkv.zF
# # #                 self.q_score.j << self.Qqkv_score.outputs

# # #                 self.Qscore_fc1.inputs << self.q_score.zF
# # #                 self.q_fc1.j << self.Qscore_fc1.outputs

# # #                 self.Qfc1_fc2.inputs << self.q_fc1.zF
# # #                 self.q_fc2.j << self.Qfc1_fc2.outputs

# # #                 self.Qfc2_out.inputs << self.q_fc2.zF
# # #                 self.q_out.j << self.Qfc2_out.outputs

# # #                 self.Qout_target.inputs << self.q_out.zF
# # #                 self.qtarget_logits.j << self.Qout_target.outputs



# # #                 # wire self.Qout_target.outputs to 
# #                 self.e_Qtarget.target << self.qtarget_logits.z
             

# #                 advance_process = (
# #     JaxProcess(name="advance_process")
# #     >> self.Efc1_score.advance_state
# #     >> self.Efc2_fc1.advance_state
# #     >> self.Eout_fc2.advance_state
# #     >> self.Etarget_out.advance_state
# #     >> self.z_qkv.advance_state
# #     >> self.z_score.advance_state
# #     >> self.z_fc1.advance_state
# #     >> self.z_fc2.advance_state
# #     >> self.zout.advance_state
# #     >> self.target_logits.advance_state
# #     >> self.Wqkv_score.advance_state
# #     >> self.Wscore_fc1.advance_state
# #     >> self.Wfc1_fc2.advance_state
# #     >> self.Wfc2_zout.advance_state
# #     >> self.Wout_target.advance_state
# #     >> self.e_score.advance_state
# #     >> self.e_fc1.advance_state
# #     >> self.e_fc2.advance_state
# #     >> self.e_zout.advance_state
# #     >> self.e_target_logits.advance_state
# # )


# #                 reset_process = (
# #     JaxProcess(name="reset_process")
# #     >> self.z_qkv.reset
# #     >> self.z_score.reset
# #     >> self.z_fc1.reset
# #     >> self.z_fc2.reset
# #     >> self.zout.reset
# #     >> self.target_logits.reset
# #     >> self.e_Qtarget.reset
# #     >> self.q_qkv.reset
# #     >> self.q_score.reset
# #     >> self.q_fc1.reset
# #     >> self.q_fc2.reset
# #     >> self.q_out.reset
# #     > self.qtarget_logits
# #     >> self.e_score.reset
# #     >> self.e_fc1.reset
# #     >> self.e_fc2.reset
# #     >> self.e_zout.reset
# #     >> self.e_target_logits.reset
# # )
# #                 evolve_process = (
# #     JaxProcess(name="evolve_process")
# #     >> self.Wqkv_score.evolve
# #     >> self.Wscore_fc1.evolve
# #     >> self.Wfc1_fc2.evolve
# #     >> self.Wfc2_zout.evolve
# #     >> self.Wout_target.evolve
# # )
# #                 project_process = (
# #     JaxProcess(name="project_process")
# #     >> self.q_qkv.advance_state
# #     >> self.Qqkv_score.advance_state
# #     >> self.q_score.advance_state
# #     >> self.Qscore_fc1.advance_state
# #     >> self.q_fc1.advance_state
# #     >> self.Qfc1_fc2.advance_state
# #     >> self.q_fc2.advance_state
# #     >> self.Qfc2_out.advance_state
# #     >> self.q_out.advance_state
# #     >> self.Qout_target.advance_state
# #     >> self.qtarget_logits.advance_state
# #     >> self.e_Qtarget.advance_state
# # )

# #                 processes = (reset_process, advance_process, evolve_process, project_process)

# #                 self._dynamic(processes)
# #     def _dynamic(self, processes):  # create dynamic commands for transformer circuit
# #         vars = self.circuit.get_components(
# #             # Rate cells
# #             "q_qkv", "q_score", "q_fc1", "q_fc2", "q_out", "qtarget_logits","e_Qtarget"

# #             # Error cells
# #             "e_score", "e_fc1", "e_fc2", "e_zout", "e_target_logits",

# #             # Projection StaticSynapses
# #             "Qqkv_score", "Qscore_fc1", "Qfc1_fc2", "Qfc2_out", "Qout_target",

# #             # Hebbian forward synapses
# #             "Wqkv_score", "Wscore_fc1", "Wfc1_fc2", "Wfc2_zout", "Wout_target",

# #             # Error synapse adapters
# #             "Efc1_score", "Efc2_fc1", "Eout_fc2", "Etarget_out",

# #             # Main dynamical z-states
# #             "z_qkv", "z_score", "z_fc1", "z_fc2", "zout", "target_logits"
# #         )
# #         (self.q_qkv, self.q_score, self.q_fc1, self.q_fc2, self.q_out, self.qtarget_logits, self.e_Qtarget,
# #         self.e_score, self.e_fc1, self.e_fc2, self.e_zout, self.e_target_logits,
# #         self.Qqkv_score, self.Qscore_fc1, self.Qfc1_fc2, self.Qfc2_out, self.Qout_target,
# #         self.Wqkv_score, self.Wscore_fc1, self.Wfc1_fc2, self.Wfc2_zout, self.Wout_target,
#         self.Efc1_score, self.Efc2_fc1, self.Eout_fc2, self.Etarget_out,
#         self.z_qkv, self.z_score, self.z_fc1, self.z_fc2, self.zout, self.target_logits) = vars

#         self.nodes = vars
        

#         self.nodes = vars

#         reset_proc, advance_proc, evolve_proc, project_proc = processes

#         # Add commands to the circuit
#         self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
#         self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
#         self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
#         self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")


#     def save_to_disk(self, params_only=False):
#         """
#         Saves current model parameter values to disk

#         Args:
#             params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
#         """
#         if params_only:
#             model_dir = "{}/{}/custom".format(self.exp_dir, self.model_name)
#             self.Wqkv_score.save(model_dir)
#             self.Wscore_fc1.save(model_dir)
#             self.Wfc1_fc2.save(model_dir)
#             self.Wfc2_zout.save(model_dir)
#             self.Wout_target.save(model_dir)

#         else:
#             self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
#             #self.circuit.save_to_json(self.exp_dir, self.model_name)

#     def load_from_disk(self, model_directory):
#         """
#         Loads parameter/config values from disk to this model

#         Args:
#             model_directory: directory/path to saved model parameter/config values
#         """
#         print(" > Loading model from ",model_directory)
#         with Context("Circuit") as self.circuit:
#             self.circuit.load_from_dir(model_directory)
#             processes = (
#                 self.circuit.reset_process, self.circuit.advance_process,
#                 self.circuit.evolve_process, self.circuit.project_process
#             )
#             self._dynamic(processes)

#     def process(self,obs,lab,adapt_synapses=True):
#         # projection,expectation,maximization
#         eps=0.001
#         _lab=jnp.clip(lab,eps,1-eps)
#         self.circuit.reset()
#         self.Qqkv_score.weights.set(self.Wqkv_score.weights.value)
#         self.Qqkv_score.biases.set(self.Wqkv_score.biases.value)

#         self.Qscore_fc1.weights.set(self.Wscore_fc1.weights.value)
#         self.Qscore_fc1.biases.set(self.Wscore_fc1.biases.value)

#         self.Qfc1_fc2.weights.set(self.Wfc1_fc2.weights.value)
#         self.Qfc1_fc2.biases.set(self.Wfc1_fc2.biases.value)

#         self.Qfc2_out.weights.set(self.Wfc2_zout.weights.value)
#         self.Qfc2_out.biases.set(self.Wfc2_zout.biases.value)

#         self.Qout_target.weights.set(self.Wout_target.weights.value)
#         self.Qout_target.biases.set(self.Wout_target.biases.value)

#         # feedback synapis 
#         # --- pin/tie feedback/error synapses to transpose of forward Hebbian synapses ---
#         self.Efc1_score.weights.set(jnp.transpose(self.Wscore_fc1.weights.value))
#         self.Efc2_fc1.weights.set(jnp.transpose(self.Wfc1_fc2.weights.value))
#         self.Eout_fc2.weights.set(jnp.transpose(self.Wfc2_zout.weights.value))
#         self.Etarget_out.weights.set(jnp.transpose(self.Wout_target.weights.value))

#         # perfom P-step (projection step)
#         self.circuit.clamp_input(obs)
#         self.circuit.clamp_infer_target(_lab)
#         self.circuit.project(t=0.,dt=1.)  # do projection/inference

#         #initialize dynamics of generative model latents to projected states
#         self.z_score.z.set(self.q_score)
#         self.z_fc1.z.set(self.q_fc1)
#         self.z_fc2.z.set(self.q_fc2)
#         self.zout.z.set(self.q_out)

#         # Note error escore=0 efc1=0 efc2=0 ezout=0 at initial states
#         self.e_target_logits.dmu.set(self.e_Qtarget.dmu.value)
#         self.e_target_logits.dtarget.set(self.e_Qtarget.dtarget.value)
#         y_mu_inf=self.qtarget_logits.z

#         EFE=0
#         y_mu=0
#         if adapt_synapses:
#             for ts in range(0,self.T):
#                 self.circuit.clamp_input(obs)
#                 self.circuit.clamp_target(_lab)
#                 self.circuit.advance(t=ts,dt=1.)

#             y_mu=self.e_target_logits.mu.value
#             L1=self.e_score.L.value
#             L2=self.e_fc1.L.value
#             L3=self.e_fc2.L.value
#             L4=self.e_zout.L.value
#             L5=self.e_target_logits.L.value
#             EFE=L1+L2+L3+L4+L5

#             if adapt_synapses==True:
#                 self.circuit.evolve(t=self.T,dt=1.)
#         return y_mu_inf,y_mu,EFE
        

        
