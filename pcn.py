from Dataloader import vocab_size
from ngcsimlib.compilers import compile_command, wrap_command
from ngclearn.utils.io_utils import makedir
from ngclearn.utils.io_utils import makedir
from ngcsimlib.context import Context
from ngclearn.utils import JaxProcess
import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out,softmax,gelu,layer_normalize
from EmbeddingHebbiansynapsis import EmbeddingHebbianSynapse
from ngclearn.utils.optim import adam
from jax import jit,random,numpy as jnp 
import jax
from ngcsimlib.utils import get_current_context
from MLP import MLP
from AttentionHebbiansynapsis import AttentionHebbianSynapse
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
# from ngclearn.com.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from flax import nnx
from flax import linen as nn
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from functools import partial as bind




class PCN():
    def __init__(self, dkey, block_size,dim,n_embed,batch_size,n_heads,dropout_rate,T=10,
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

                self.Attention_Hebbian= AttentionHebbianSynapse(
    "Attention_Hebbian",
    block_size=10,
    batch_size=4,
    vocab_size=30,
    embed_dim=128,
    shape=(2, 3),
    eta=0.0004,
    optim_type='adam',
    sign_value=-1.0,
    prior=("l1l2", 0.001),
    prior_lmbda=(0.001, 0.01), dropout_rate=dropout_rate
     # (l2_strength, l1_ratio)
)
                self.zEmbedding = RateCell("zEmbedding", n_units=dim, tau_m=0., act_fx="identity")
                self.zqkv = RateCell("zqkv", n_units=dim, tau_m=tau_m, act_fx=act_fx, prior=("gaussian", 0.), integration_type="euler")
                self.zqkv_error = ErrorCell("zqkv_error", n_units=dim)
             
                self.EmbddingHebbain = EmbeddingHebbianSynapse(
    "EmbddingHebbain",
    block_size=10,
    batch_size=4,
    vocab_size=30,
    embed_dim=128,
    shape=(2, 3),
    eta=0.0004,
    optim_type='adam',
    sign_value=-1.0,
    prior=("l1l2", 0.001),
    prior_lmbda=(0.001, 0.01),
)
# )
                self.z_out_error=ErrorCell("z_out_error", n_units=dim)
                self.z_out=RateCell("z_out",n_units=dim,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type='euler')

                self.z_score_Error=ErrorCell("z_score_Error", n_units=dim)
                self.z_score=RateCell("z_score",n_units=dim,tau_m=tau_m,act_fx="relu",prior=("gaussian",0.),integration_type='euler')



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
                "W_zqkv_q", shape=(dim,dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                            bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                

                self.W_zqkv_k=HebbianSynapse(
                "W_zqkv_k", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.W_zqkv_v=HebbianSynapse(
                "W_zqkv_v", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Attention_to_mlp = HebbianSynapse(
                    "Attention_to_mlp", shape=(dim, dim), eta=eta, weight_init=dist.uniform(amin=wlb, amax=wub),
                    bias_init=dist.constant(value=0.), w_bound=0., optim_type=optim_type, sign_value=-1., key=subkeys[4]
                )
                self.Emlp1_to_attention = StaticSynapse(
                            "Emlp1_to_attention", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                        )
                self.Ezscore_zqkv = StaticSynapse(
                            "Ezscore_zqkv", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
                        )

                

                # forward connection
                # TODO CHECK IF EmbeddingHebbian works 
                self.EmbddingHebbain.inputs << self.zEmbedding.zF
                self.zqkv_error.mu<< self.EmbddingHebbain.outputs
                self.zqkv_error.target << self.zqkv.z

                # adding qkv to the attention 
                self.W_zqkv_q.inputs << self.zqkv.zF
                self.Attention_Hebbian.inputs_q << self.W_zqkv_q.outputs
                self.W_zqkv_k.inputs << self.zqkv.zF
                self.Attention_Hebbian.inputs_k << self.W_zqkv_k.outputs
                self.W_zqkv_v.inputs << self.zqkv.zF
                self.Attention_Hebbian.inputs_v << self.W_zqkv_v.outputs
                self.z_score_Error.mu <<  self.Attention_Hebbian.outputs
                self.z_score_Error.target << self.z_score.z
    

                self.Attention_to_mlp.inputs << self.z_score.zF
                self.MLP.mlp_1_error.mu << self.Attention_to_mlp.outputs
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
        
                self.Ezscore_zqkv.inputs << self.z_score_Error.dmu
                self.zqkv.j<< self.Ezscore_zqkv.outputs
                self.zqkv.j_td << self.zqkv_error.dtarget

               

                # feedback attention out  mlp 1
                self.MLP.Emlp1_to_z_score.inputs << self.MLP.mlp_1_error.dmu
                self.z_score.j << self.MLP.Emlp1_to_z_score.outputs
                self.z_score.j_td << self.z_score_Error.dtarget


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
      


                self.EmbddingHebbain.pre << self.zEmbedding.zF
                self.EmbddingHebbain.post << self.zqkv_error.dmu





                self.Attention_Hebbian.pre << self.zqkv.zF
                self.Attention_Hebbian.post << self.z_score_Error.dmu # if needed


                self.Attention_to_mlp.pre << self.z_score.zF
                self.Attention_to_mlp.post << self.MLP.mlp_1_error.dmu     # if needed

                
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
              
                self.q_score = RateCell("q_score", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_mlp1 = RateCell("q_mlp1", n_units=4 *dim, tau_m=0., act_fx=act_fx)
                self.q_mlp2 = RateCell("q_mlp2", n_units=dim, tau_m=0., act_fx=act_fx)
                self.q_out = RateCell("q_out", n_units=dim, tau_m=0., act_fx=act_fx)
                self.qtarget_logits = RateCell("qtarget_logits", n_units=dim, tau_m=0., act_fx="identity")
                self.qError_target_logit=ErrorCell("qError_target_logit", n_units=dim)
#                 # --- ERROR CELL ---
                self.e_Qtarget = ErrorCell("e_Qtarget", n_units=dim)

              
                # --- STATIC SYNAPSES ---
                
                self.Qembed_zqkv = StaticSynapse(
                    "Qembed_zqkv", shape=(dim, dim),
                    bias_init=dist.constant(value=0.), key=subkeys[0]
                )
                
                
                self.Qqkv_score = StaticSynapse(
                    "Qqkv_score", shape=(dim, dim),
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

                self.Qqkv_score.inputs << self.q_zqkv.zF
                self.q_score.j<< self.Qqkv_score.outputs

                # Attention → MLP1
                self.QAttention_mlp1.inputs << self.q_score.zF
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
>> self.zEmbedding.advance_state
>> self.zqkv_error.advance_state
>> self.zqkv.advance_state
>> self.W_zqkv_q.advance_state
>> self.Attention_Hebbian.advance_state_attention
>> self.W_zqkv_k.advance_state
>> self.W_zqkv_v.advance_state
>> self.z_score_Error.advance_state
>> self.z_score.advance_state
>> self.Attention_to_mlp.advance_state
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
>> self.Ezscore_zqkv.advance_state
>> self.MLP.Emlp1_to_z_score.advance_state
>> self.MLP.Emlp2_mlp1.advance_state
>> self.MLP.Ezout_mlp2.advance_state
>> self.Etargetlogit_zout.advance_state
>> self.EmbddingHebbain.advance_state_hebbian


                )





                reset_process = (JaxProcess(name="reset_process")
>> self.zEmbedding.reset
>> self.zqkv_error.reset
>> self.zqkv.reset
>> self.W_zqkv_q.reset
>> self.W_zqkv_k.reset
>> self.W_zqkv_v.reset
>> self.z_score_Error.reset
>> self.z_score.reset
>> self.MLP.mlp_1_error.reset
>> self.MLP.mlp_1.reset
>> self.MLP.mlp1_mlp2.reset
>> self.MLP.mlp_2_error.reset
>> self.MLP.mlp_2.reset
>> self.z_out_error.reset
>> self.z_out.reset
>> self.target_logit_error.reset
>> self.target_logit.reset
>> self.zqkv.reset
>> self.q_embed.reset
>> self.q_zqkv.reset
>> self.q_score.reset
>> self.q_mlp1.reset
>> self.q_mlp2.reset
>> self.q_out.reset
>> self.qtarget_logits.reset
>> self.qError_target_logit.reset



                        )


                # # evolve_process = (
                #         JaxProcess(name="evolve_process")
                #         >> self.EmbddingHebbain.evolve
                #         >> self.zout_targetlogit.evolve
                #         >> self.Attention_Hebbian.evolve
                #         >> self.zout_targetlogit.evolve
                #         >> self.Attention_to_mlp.evolve




                    # )

                project_process = (
                    JaxProcess(name="project_process")
                
                            >> self.q_embed.advance_state
                            >> self.q_zqkv.advance_state
                            >> self.q_score.advance_state
                            >> self.q_mlp1.advance_state
                            >> self.q_mlp2.advance_state
                            >> self.q_out.advance_state
                            >> self.qtarget_logits.advance_state
                            >> self.qError_target_logit.advance_state
                            >> self.Qembed_zqkv.advance_state
                            >> self.Qqkv_score.advance_state
                            >> self.QAttention_mlp1.advance_state
                            >> self.Qmlp1_mlp2.advance_state
                            >> self.Qmlp2_out.advance_state
                            >> self.Qout_target.advance_state



                   
                )
                process = (reset_process, advance_process,project_process)

                self._dynamic(process)
    def _dynamic(self,process):
        vars = self.circuit.get_components( 
          "zEmbedding",
"zqkv_error",
"zqkv",
"W_zqkv_q",
"Attention_Hebbian",
"W_zqkv_k",
"W_zqkv_v",
"z_score_Error",
"z_score",
"Attention_to_mlp",
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
"Ezscore_zqkv",
"Emlp1_to_z_score",
"Emlp2_mlp1",
"Ezout_mlp2",
"Etargetlogit_zout",
"EmbddingHebbain",
"q_embed",
"q_zqkv",
"q_score",
"q_mlp1",
"q_mlp2",
"q_out",
"qtarget_logits",
"qError_target_logit",
"Qembed_zqkv",
"Qqkv_score",
"QAttention_mlp1",
"Qmlp1_mlp2",
"Qmlp2_out",
"Qout_target",


                        )

                #     # # Get components
                #     #     *component_names)

                #     # Unpack into attributes
        (   
                self.zEmbedding,
                self.zqkv_error,
                self.zqkv,
                self.W_zqkv_q,
                self.Attention_Hebbian,
                self.W_zqkv_k,
                self.W_zqkv_v,
                self.z_score_Error,
                self.z_score,
                self.Attention_to_mlp,
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
                self.Ezscore_zqkv,
                self.Emlp1_to_z_score,
                self.Emlp2_mlp1,
                self.Ezout_mlp2,
                self.Etargetlogit_zout,
                self.EmbddingHebbain,
                self.q_embed,
                self.q_zqkv,
                self.q_score,
                self.q_mlp1,
                self.q_mlp2,
                self.q_out,
                self.qtarget_logits,
                self.qError_target_logit,
                self.Qembed_zqkv,
                self.Qqkv_score,
                self.QAttention_mlp1,
                self.Qmlp1_mlp2,
                self.Qmlp2_out,
                self.Qout_target,

        ) = vars
        self.nodes = vars

        reset_proc, advance_proc,  project_proc = process


        self.circuit.wrap_and_add_command(jit(reset_proc.pure), name="reset")
        self.circuit.wrap_and_add_command(jit(advance_proc.pure), name="advance")
        self.circuit.wrap_and_add_command(jit(project_proc.pure), name="project")
        # self.circuit.wrap_and_add_command(jit(evolve_proc.pure), name="evolve")
        @Context.dynamicCommand
        def clamp_input(x):
            self.zEmbedding.j.set(x)
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

            self.inputs_k_z_score.save(model_dir)
            self.inputs_q_z_score.save(model_dir)
            self.inputs_v_z_score.save(model_dir)

            self.Attention_Hebbian_to_mlp.save(model_dir)

            self.mlp1_mlp2.save(model_dir)
            self.mlp2_zout.save(model_dir)

            self.zout_targetlogit.save(model_dir)

    #     else:
    #         self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
    # def load_from_disk(self, model_directory):
    #     """
    #     Loads parameter/config values from disk to this model

    #     Args:
    #         model_directory: directory/path to saved model parameter/config values
    #     """
    #     print(" > Loading model from ",model_directory)
    #     with Context("Circuit") as self.circuit:
    #         self.circuit.load_from_dir(model_directory)
    #         processes = (
    #             self.circuit.reset_process, self.circuit.advance_process,
    #             self.circuit.evolve_process, self.circuit.project_process
    #         )
    #         self._dynamic(processes)

    def process(self, obs, lab, adapt_synapses=True):
        print("obs shape",{obs.shape})
        print("lab shape",{lab.shape})
        

        ## can think of the PCN as doing "PEM" -- projection, expectation, then maximization
        eps = 0.001
        _lab = jnp.clip(lab, eps, 1. - eps)
        # self.circuit.reset(do_reset=True)
        self.circuit.reset()
       
        ## pin/tie inference synapses to be exactly equal to the forward ones
        
       # Copy encoder/decoder weights between W and Q pathways
        self.Qembed_zqkv.weights.set(self.EmbddingHebbain.weights.value)
        self.Qembed_zqkv.biases.set(self.EmbddingHebbain.biases.value)

        self.Qqkv_score.weights.set(self.Attention_Hebbian.weights.value)
        self.Qqkv_score.biases.set(self.Attention_Hebbian.biases.value)

        self.QAttention_mlp1.weights.set(self.Attention_to_mlp.weights.value)
        self.QAttention_mlp1.biases.set(self.Attention_to_mlp.biases.value)

        self.Qmlp1_mlp2.weights.set(self.MLP.mlp1_mlp2.weights.value)
        self.Qmlp1_mlp2.biases.set(self.MLP.mlp1_mlp2.biases.value)

        self.Qmlp2_out.weights.set(self.MLP.mlp2_zout.weights.value)
        self.Qmlp2_out.biases.set(self.MLP.mlp2_zout.biases.value)

        self.Qout_target.weights.set(self.zout_targetlogit.weights.value)
        self.Qout_target.biases.set(self.zout_targetlogit.biases.value)


        ## pin/tie feedback synapses to transpose of forward ones
        # self.Ezscore_zqkv.weights.set(jnp.transpose(self.EmbddingHebbain.weights.value))
        

        self.Ezscore_zqkv.weights.set(jnp.transpose(self.Attention_Hebbian.weights.value))
        

        self.MLP.Emlp1_to_z_score.weights.set(jnp.transpose(self.Attention_to_mlp.weights.value))
        

        self.MLP.Emlp2_mlp1.weights.set(jnp.transpose(self.MLP.mlp1_mlp2.weights.value))
    

        self.MLP.Ezout_mlp2.weights.set(jnp.transpose(self.MLP.mlp2_zout.weights.value))
        

        self.Etargetlogit_zout.weights.set(jnp.transpose(self.zout_targetlogit.weights.value))
        

        ## Perform P-step (projection step)
        self.circuit.clamp_input(obs)
        self.circuit.clamp_infer_target(_lab)
        
        self.circuit.project(t=0., dt=1.) ## do projection/inference
        
        ## initialize dynamics of generative model latents to projected states
        self.zqkv.z.set(self.q_zqkv.z.value)

        self.z_score.z.set(self.q_score.z.value)
        self.MLP.mlp_1.z.set(self.q_mlp1.z.value)
        self.MLP.mlp_2.z.set(self.q_mlp2.z.value)
        self.z_out.z.set(self.q_out.z.value)

        # ## self.z3.z.set(self.q3.z.value)
        # # ### Note: e1 = 0, e2 = 0 at initial conditions
        self.target_logit_error.dmu.set(self.qError_target_logit.dmu.value)
        self.target_logit_error.dtarget.set(self.qError_target_logit.dtarget.value)
        # ## get projected prediction (from the P-step)
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
          
            L1 = self.zqkv_error.L.value
            L2 = self.z_score_Error.L.value
            L3 = self.MLP.mlp_1_error.L.value
            L4 = self.MLP.mlp_2_error.L.value
            L5 = self.z_out_error.L.value
            L6 = self.target_logit_error.L.value

            EFE = L1 + L2 + L3 + L4 + L5 + L6 


            # ## Perform (optional) M-step (scheduled synaptic updates)
            # if adapt_synapses == True:
            #     #self.circuit.evolve(t=self.T, dt=self.dt)
            #     self.circuit.evolve(t=self.T, dt=1.)
        ## skip E/M steps if just doing test-time inference
        print(f"y_mu_inf -------------"  ,{y_mu_inf})
        print(f"y_mu -------------------- ",{y_mu})
        print(f"EFE ________________________",{EFE})
        return y_mu_inf, y_mu, EFE

    # # def get_latents(self):
    # #     return self.q2.z.value

    # def _get_norm_string(self):  ## debugging routine
    # # weights
    #     W_zqkv_q = self.W_zqkv_q.weights.value
    #     W_zqkv_k = self.W_zqkv_k.weights.value
    #     W_zqkv_v = self.W_zqkv_v.weights.value
    #     inputs_k_z_score = self.inputs_k_z_score.weights.value
    #     inputs_q_z_score = self.inputs_q_z_score.weights.value
    #     inputs_v_z_score = self.inputs_v_z_score.weights.value
    #     attention_to_mlp = self.Attention_Hebbian_to_mlp.weights.value
    #     mlp1_mlp2 = self.mlp1_mlp2.weights.value
    #     mlp2_zout = self.mlp2_zout.weights.value
    #     zout_targetlogit = self.zout_targetlogit.weights.value

    #     # biases (if they exist)
    #     b_emb_q = self.W_zqkv_q.biases.value
    #     b_emb_k = self.W_zqkv_k.biases.value
    #     b_emb_v = self.W_zqkv_v.biases.value
    #     b_inputs_k_z_score = self.inputs_k_z_score.biases.value
    #     b_inputs_q_z_score = self.inputs_q_z_score.biases.value
    #     b_inputs_v_z_score = self.inputs_v_z_score.biases.value
    #     b_attention_to_mlp = self.Attention_Hebbian_to_mlp.biases.value
    #     b_mlp1_mlp2 = self.mlp1_mlp2.biases.value
    #     b_mlp2_zout = self.mlp2_zout.biases.value
    #     b_zout_targetlogit = self.zout_targetlogit.biases.value

    #     _norms = (
    #         f"W_zqkv_q: {jnp.linalg.norm(W_zqkv_q)} "
    #         f"W_zqkv_k: {jnp.linalg.norm(W_zqkv_k)} "
    #         f"W_zqkv_v: {jnp.linalg.norm(W_zqkv_v)} "
    #         f"inputs_k_z_score: {jnp.linalg.norm(inputs_k_z_score)} "
    #         f"inputs_q_z_score: {jnp.linalg.norm(inputs_q_z_score)} "
    #         f"inputs_v_z_score: {jnp.linalg.norm(inputs_v_z_score)} "
    #         f"attention_to_mlp: {jnp.linalg.norm(attention_to_mlp)} "
    #         f"mlp1_mlp2: {jnp.linalg.norm(mlp1_mlp2)} "
    #         f"mlp2_zout: {jnp.linalg.norm(mlp2_zout)} "
    #         f"zout_targetlogit: {jnp.linalg.norm(zout_targetlogit)}\n"
    #         f"b_emb_q: {jnp.linalg.norm(b_emb_q)} "
    #         f"b_emb_k: {jnp.linalg.norm(b_emb_k)} "
    #         f"b_emb_v: {jnp.linalg.norm(b_emb_v)} "
    #         f"b_inputs_k_z_score: {jnp.linalg.norm(b_inputs_k_z_score)} "
    #         f"b_inputs_q_z_score: {jnp.linalg.norm(b_inputs_q_z_score)} "
    #         f"b_inputs_v_z_score: {jnp.linalg.norm(b_inputs_v_z_score)} "
    #         f"b_attention_to_mlp: {jnp.linalg.norm(b_attention_to_mlp)} "
    #         f"b_mlp1_mlp2: {jnp.linalg.norm(b_mlp1_mlp2)} "
    #         f"b_mlp2_zout: {jnp.linalg.norm(b_mlp2_zout)} "
    #         f"b_zout_targetlogit: {jnp.linalg.norm(b_zout_targetlogit)}"
    #     )
        
    #     return _norms

        