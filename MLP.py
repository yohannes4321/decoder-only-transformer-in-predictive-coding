import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse


class MLP(JaxComponent):
    def __init__(self, name, dkey, dim, act_fx, tau_m, eta, wlb, wub, optim_type):
        super().__init__(name)

        # ✅ Always split keys first
        dkey, *subkeys = random.split(dkey, 10)
        # (we will use k1 and k2)

        # ---------------------
        # Layer 1
        # ---------------------
        self.mlp_1_error = ErrorCell("mlp_1_error", n_units=4 * dim)

        self.mlp_1 = RateCell(
            "mlp_1",
            n_units=4 * dim,
            tau_m=tau_m,
            act_fx="relu",
            prior=("gaussian", 0.),
            integration_type='euler'
        )

        self.mlp1_mlp2 = HebbianSynapse(
            "Wdim",
            shape=(4 * dim, dim),
            eta=eta,
            weight_init=dist.uniform(amin=wlb, amax=wub),
            bias_init=dist.constant(0.),
            w_bound=0.,
            optim_type=optim_type,
            sign_value=-1.,
            key=subkeys[4]            # ✅ unique key
        )

        # ---------------------
        # Layer 2
        # ---------------------
        self.mlp_2 = RateCell(
            "mlp_2",
            n_units=dim,
            tau_m=tau_m,
            act_fx=act_fx,
            prior=("gaussian", 0.),
            integration_type='euler'
        )

        self.mlp_2_error = ErrorCell("mlp_2_error", n_units=dim)

        self.mlp_zout = HebbianSynapse(
            "mlp_zout",
            shape=(dim, dim),
            eta=eta,
            weight_init=dist.uniform(amin=wlb, amax=wub),
            bias_init=dist.constant(0.),
            w_bound=0.,
            optim_type=optim_type,
            sign_value=-1.,
            key=subkeys[4]          # ✅ another unique key
        )
