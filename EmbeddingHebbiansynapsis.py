from jax import random, numpy as jnp, jit
from functools import partial
from ngclearn.utils.optim import get_opt_init_fn, get_opt_step_fn
from ngclearn import resolver, Component, Compartment
from ngcsimlib.compilers.process import transition
from ngclearn.components.synapses import DenseSynapse
from ngclearn.utils import tensorstats
from ngcsimlib.deprecators import deprecate_args
import jax
import jax.numpy as jnp
from flax import nnx
@partial(jit, static_argnums=[3, 4, 5, 6, 7, 8, 9])
def _calc_update(pre, post, W, w_bound, is_nonnegative=True, signVal=1.,
                 prior_type=None, prior_lmbda=0.,
                 pre_wght=1., post_wght=1.):
    """
    Compute a tensor of adjustments to be applied to a synaptic value matrix.

    Args:
        pre: pre-synaptic statistic to drive Hebbian update

        post: post-synaptic statistic to drive Hebbian update

        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: (Unused)

        signVal: multiplicative factor to modulate final update by (good for
            flipping the signs of a computed synaptic change matrix)

        prior_type: prior type or name (Default: None)

        prior_lmbda: prior parameter (Default: 0.0)

        pre_wght: pre-synaptic weighting term (Default: 1.)

        post_wght: post-synaptic weighting term (Default: 1.)

    Returns:
        an update/adjustment matrix, an update adjustment vector (for biases)
    """
    _pre = pre * pre_wght
    _post = post * post_wght
    dW = jnp.matmul(_pre.T, _post) ## calc Hebbian adjustment
    db = jnp.sum(_post, axis=0, keepdims=True) ## calc Hebbian adjustment to bias/base-rates
    dW_reg = 0. ## synaptic decay term

    if w_bound > 0.: ## induce any synaptic value bounding
        dW = dW * (w_bound - jnp.abs(W))
    ## apply synaptic priors
    if prior_type == "l2" or prior_type == "ridge":
        dW_reg = -W * prior_lmbda
    if prior_type == "l1" or prior_type == "lasso":
        dW_reg = -jnp.sign(W) * prior_lmbda
    if prior_type == "l1l2" or prior_type == "elastic_net":
        l1_ratio = prior_lmbda[1]
        prior_scale = prior_lmbda[0]
        dW_reg = -jnp.sign(W) * l1_ratio - W * (1-l1_ratio)/2
        dW_reg = dW_reg * prior_scale
    ## produce final update/adjustment
    dW = dW + dW_reg
    return dW * signVal, db * signVal

@partial(jit, static_argnums=[1,2])
def _enforce_constraints(W, w_bound, is_nonnegative=True):
    """
    Enforces constraints that the (synaptic) efficacies/values within matrix
    `W` must adhere to.

    Args:
        W: synaptic weight values (at time t)

        w_bound: maximum value to enforce over newly computed efficacies

        is_nonnegative: ensure updated value matrix is strictly non-negative

    Returns:
        the newly evolved synaptic weight value matrix
    """
    _W = W
    if w_bound > 0.:
        if is_nonnegative == True:
            _W = jnp.clip(_W, 0., w_bound)
        else:
            _W = jnp.clip(_W, -w_bound, w_bound)
    return _W


class EmbeddingHebbianSynapse(DenseSynapse):
    """
    A synaptic cable that adjusts its efficacies via a two-factor Hebbian
    adjustment rule.

    | --- Synapse Compartments: ---
    | inputs - input (takes in external signals)
    | outputs - output signals (transformation induced by synapses)
    | weights - current value matrix of synaptic efficacies
    | biases - current value vector of synaptic bias values
    | key - JAX PRNG key
    | --- Synaptic Plasticity Compartments: ---
    | pre - pre-synaptic signal to drive first term of Hebbian update (takes in external signals)
    | post - post-synaptic signal to drive 2nd term of Hebbian update (takes in external signals)
    | dWeights - current delta matrix containing changes to be applied to synaptic efficacies
    | dBiases - current delta vector containing changes to be applied to bias values
    | opt_params - locally-embedded optimizer statisticis (e.g., Adam 1st/2nd moments if adam is used)

    Args:
        name: the string name of this cell

        shape: tuple specifying shape of this synaptic cable (usually a 2-tuple
            with number of inputs by number of outputs)

        eta: global learning rate

        weight_init: a kernel to drive initialization of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            initialization to use

        bias_init: a kernel to drive initialization of biases for this synaptic cable
            (Default: None, which turns off/disables biases)

        w_bound: maximum weight to softly bound this cable's value matrix to; if
            set to 0, then no synaptic value bounding will be applied

        is_nonnegative: enforce that synaptic efficacies are always non-negative
            after each synaptic update (if False, no constraint will be applied)

        prior: a kernel to drive prior of this synaptic cable's values;
            typically a tuple with 1st element as a string calling the name of
            prior to use and 2nd element as a floating point number
            calling the prior parameter lambda (Default: ('constant', 0.))
            currently it supports "l1"/"lasso"/"laplacian" or "l2"/"ridge"/"gaussian" or "l1l2"/"elastic_net".
            usage guide:
            prior = ('l1', 0.01) or prior = ('lasso', lmbda)
            prior = ('l2', 0.01) or prior = ('ridge', lmbda)
            prior = ('l1l2', (0.01, 0.01)) or prior = ('elastic_net', (lmbda, l1_ratio))

        sign_value: multiplicative factor to apply to final synaptic update before
            it is applied to synapses; this is useful if gradient descent style
            optimization is required (as Hebbian rules typically yield
            adjustments for ascent)

        optim_type: optimization scheme to physically alter synaptic values
            once an update is computed (Default: "sgd"); supported schemes
            include "sgd" and "adam"

            :Note: technically, if "sgd" or "adam" is used but `signVal = 1`,
                then the ascent form of each rule is employed (signVal = -1) or
                a negative learning rate will mean a descent form of the
                `optim_scheme` is being employed

        pre_wght: pre-synaptic weighting factor (Default: 1.)

        post_wght: post-synaptic weighting factor (Default: 1.)

        resist_scale: a fixed scaling factor to apply to synaptic transform
            (Default: 1.), i.e., yields: out = ((W * Rscale) * in) + b

        p_conn: probability of a connection existing (default: 1.); setting
            this to < 1. will result in a sparser synaptic structure
    """

    # Define Functions
    @deprecate_args(_rebind=False, w_decay='prior')
    def __init__(self, name,  block_size,batch_size, vocab_size, embed_dim,shape, eta=0., weight_init=None, bias_init=None,
                 w_bound=1., is_nonnegative=False, prior=("constant", 0.), w_decay=0., sign_value=1.,
                 optim_type="sgd", pre_wght=1., post_wght=1., p_conn=1.,
                 resist_scale=1., **kwargs):
        super().__init__(name, shape, weight_init, bias_init, resist_scale,
                         p_conn, batch_size=batch_size, **kwargs)

        if w_decay > 0.:
            prior = ('l2', w_decay)

        prior_type, prior_lmbda = prior
        if prior_type is None:
            prior_type = "constant"
        ## synaptic plasticity properties and characteristics
        self.shape = shape
        self.Rscale = resist_scale
        self.prior_type = prior_type
        if self.prior_type.lower() == "gaussian":
            self.prior_type = "ridge"
        elif self.prior_type.lower() == "laplacian":
            self.prior_type = "lasso"
        self.prior_lmbda = prior_lmbda
        self.w_bound = w_bound
        self.pre_wght = pre_wght
        self.post_wght = post_wght
        self.eta = eta
        self.is_nonnegative = is_nonnegative
        self.sign_value = sign_value


        ## optimization / adjustment properties (given learning dynamics above)
        self.opt = get_opt_step_fn(optim_type, eta=self.eta)

        # compartments (state of the cell, parameters, will be updated through stateless calls)
        self.preVals = jnp.zeros((self.batch_size, shape[0]))
        self.postVals = jnp.zeros((self.batch_size, shape[1]))
        self.pre = Compartment(self.preVals)
        self.post = Compartment(self.postVals)
        self.dWeights = Compartment(jnp.zeros(shape))
        self.dBiases = Compartment(jnp.zeros(shape[1]))
        self.batch_size = batch_size
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.inputs= Compartment(jnp.zeros((self.batch_size,self.block_size)))
        self.outputs= Compartment(jnp.zeros((self.batch_size*self.batch_size,self.embed_dim)))
        

        #key, subkey = random.split(self.key.value)
        self.opt_params = Compartment(get_opt_init_fn(optim_type)(
            [self.weights.value, self.biases.value]
            if bias_init else [self.weights.value]))
    



    @staticmethod
    def token_positional_embedding(x, vocab_size, embed_dim, block_size, dropout_rate):
        """
        x: [batch_size, seq_len] token indices
        returns: [batch_size * seq_len, embed_dim]
        """
        key = jax.random.PRNGKey(0)
        rngs = nnx.Rngs(key)
        wte = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
        wpe = nnx.Embed(num_embeddings=block_size, features=embed_dim, rngs=rngs)
        dropout = nnx.Dropout(rate=dropout_rate, rngs=rngs)

        batch_size, seq_len = x.shape
        tok_emb = wte(x)  # (B, T, C)
        pos = jnp.arange(seq_len)[None, :]  # (1, T)
        pos_emb = wpe(pos)
        drive = dropout(tok_emb + pos_emb)
        drive_2d = drive.reshape(batch_size * seq_len, embed_dim)
        return drive_2d

    # class EmbeddingHebbianSynapse(DenseSynapse):
    @transition(output_compartments=["inputs","outputs"])
    @staticmethod
    def advance_state_hebbian(inputs, weights, pre, post, vocab_size, embed_dim, block_size,dropout_rate,
                              w_bounds=1.0, w_decay=0.0, eta=0.001):
        """
        Custom Hebbian update + embedding computation (standalone, no @transition)
        """
        # Compute embeddings
        outputs = EmbeddingHebbianSynapse.token_positional_embedding(
            inputs, vocab_size, embed_dim, block_size,dropout_rate
        )
                # inputs, weights, pre, post, vocab_size, embed_dim, block_size,dropout_rate

        # Hebbian weight update
        dW = jnp.matmul(pre.T, post)
        weights = weights + eta * (dW - weights * w_decay)
        weights = jnp.clip(weights, 0., w_bounds)

        return outputs, weights


       
    @staticmethod
    def _compute_update(w_bound, is_nonnegative, sign_value, prior_type, prior_lmbda, pre_wght,
                        post_wght, pre, post, weights):
        
        ## calculate synaptic update values
        dW, db = _calc_update(
            pre, post, weights, w_bound, is_nonnegative=is_nonnegative,
            signVal=sign_value, prior_type=prior_type, prior_lmbda=prior_lmbda, pre_wght=pre_wght,
            post_wght=post_wght)
        return dW, db

    @transition(output_compartments=["inputs","outputs","opt_params", "weights", "biases", "dWeights", "dBiases"])
    @staticmethod
    def evolve(inputs,outputs,vocab_size,embed_dim,block_size,opt, w_bound, is_nonnegative, sign_value, prior_type, prior_lmbda, pre_wght,
                post_wght, bias_init, pre, post, weights, biases, opt_params):
        
        # tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
        
        # outputs  = EmbeddingHebbianSynapse.token_positional_embedding(inputs, vocab_size, embed_dim, block_size, rngs=rngs)
        
        




        ## calculate synaptic update values
        dWeights, dBiases = EmbeddingHebbianSynapse._compute_update(
            w_bound, is_nonnegative, sign_value, prior_type, prior_lmbda, pre_wght, post_wght,
            pre, post, weights
        )
        ## conduct a step of optimization - get newly evolved synaptic weight value matrix
        if bias_init != None:
            opt_params, [weights, biases] = opt(opt_params, [weights, biases], [dWeights, dBiases])
        else:
            # ignore db since no biases configured
            opt_params, [weights] = opt(opt_params, [weights], [dWeights])
        ## ensure synaptic efficacies adhere to constraints
        weights = _enforce_constraints(weights, w_bound, is_nonnegative=is_nonnegative)
        return inputs,outputs,opt_params, weights, biases, dWeights, dBiases

    @transition(output_compartments=["inputs", "outputs", "pre", "post", "dWeights", "dBiases"])
    @staticmethod
    def reset(batch_size, shape):
        preVals = jnp.zeros((batch_size, shape[0]))
        postVals = jnp.zeros((batch_size, shape[1]))
        return (
            preVals, # inputs
            postVals, # outputs
            preVals, # pre
            postVals, # post
            jnp.zeros(shape), # dW
            jnp.zeros(shape[1]), # db
        )

    @classmethod
    def help(cls): ## component help function
        properties = {
            "synapse_type": "EmbeddingHebbianSynapse - performs an adaptable synaptic "
                            "transformation of inputs to produce output signals; "
                            "synapses are adjusted via two-term/factor Hebbian adjustment"
        }
        compartment_props = {
            "inputs":
                {"inputs": "Takes in external input signal values",
                 "pre": "Pre-synaptic statistic for Hebb rule (z_j)",
                 "post": "Post-synaptic statistic for Hebb rule (z_i)"},
            "states":
                {"weights": "Synapse efficacy/strength parameter values",
                 "biases": "Base-rate/bias parameter values",
                 "key": "JAX PRNG key"},
            "analytics":
                {"dWeights": "Synaptic weight value adjustment matrix produced at time t",
                 "dBiases": "Synaptic bias/base-rate value adjustment vector produced at time t"},
            "outputs":
                {"outputs": "Output of synaptic transformation"},
        }
        hyperparams = {
            "shape": "Shape of synaptic weight value matrix; number inputs x number outputs",
            "batch_size": "Batch size dimension of this component",
            "weight_init": "Initialization conditions for synaptic weight (W) values",
            "bias_init": "Initialization conditions for bias/base-rate (b) values",
            "resist_scale": "Resistance level scaling factor (applied to output of transformation)",
            "p_conn": "Probability of a connection existing (otherwise, it is masked to zero)",
            "is_nonnegative": "Should synapses be constrained to be non-negative post-updates?",
            "sign_value": "Scalar `flipping` constant -- changes direction to Hebbian descent if < 0",
            "eta": "Global (fixed) learning rate",
            "pre_wght": "Pre-synaptic weighting coefficient (q_pre)",
            "post_wght": "Post-synaptic weighting coefficient (q_post)",
            "w_bound": "Soft synaptic bound applied to synapses post-update",
            "prior": "prior name and value for synaptic updating prior",
            "optim_type": "Choice of optimizer to adjust synaptic weights"
        }
        info = {cls.__name__: properties,
                "compartments": compartment_props,
                "dynamics": "outputs = [(W * Rscale) * inputs] + b ;"
                            "dW_{ij}/dt = eta * [(z_j * q_pre) * (z_i * q_post)] - g(W_{ij}) * prior_lmbda",
                "hyperparameters": hyperparams}
        return info

    def __repr__(self):
        comps = [varname for varname in dir(self) if Compartment.is_compartment(getattr(self, varname))]
        maxlen = max(len(c) for c in comps) + 5
        lines = f"[{self.__class__.__name__}] PATH: {self.name}\n"
        for c in comps:
            stats = tensorstats(getattr(self, c).value)
            if stats is not None:
                line = [f"{k}: {v}" for k, v in stats.items()]
                line = ", ".join(line)
            else:
                line = "None"
            lines += f"  {f'({c})'.ljust(maxlen)}{line}\n"
        return lines

if __name__ == '__main__':
   
    from ngcsimlib.context import Context
    with Context("Bar") as bar:
        Wab = EmbeddingHebbianSynapse(
    "Wab",
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





    batch_size = 4
    seq_len = 8
    vocab_size = 30
    embed_dim = 128
    block_size = 10
    dropout_rate=0.5

    # random inputs / pre / post
    inputs = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, vocab_size)
    pre = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 2))
    post = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 3))
    weights = jax.random.normal(jax.random.PRNGKey(3), (2, 3))

    outputs, weights = Wab.advance_state_hebbian(
    inputs=inputs, weights=weights, pre=pre, post=post, vocab_size=vocab_size, embed_dim=embed_dim, block_size=block_size, dropout_rate=dropout_rate
)

    print("input_shpae", inputs.shape)
    print("Outputs shape:", outputs.shape)  # should be (batch_size * seq_len, embed_dim)
    print("Weights shape:", weights.shape)  # should be same as original (2,3)