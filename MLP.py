import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.model_utils import drop_out, softmax, gelu, layer_normalize
from ngclearn.components.jaxComponent import JaxComponent
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse,StaticSynapse
from ngcsimlib.context import Context
from ngcsimlib.utils import get_current_context
class MLP(JaxComponent):
    def __init__(self, name, dkey, dim, act_fx, tau_m, eta, wlb, wub, optim_type):
        super().__init__(name)

        dkey, *subkeys = random.split(dkey, 10)
        # NOTE: don't create a new context here; use the one in PCN
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
            "mlp1_mlp2",
            shape=(4 * dim, dim),
            eta=eta,
            weight_init=dist.uniform(amin=wlb, amax=wub),
            bias_init=dist.constant(0.),
            w_bound=0.,
            optim_type=optim_type,
            sign_value=-1.,
            key=subkeys[4]          
        )

        self.Emlp2_mlp1 = StaticSynapse(
            "Emlp2_mlp1", shape=(dim, 4 * dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
        )
        self.Emlp1_to_attentionout = StaticSynapse(
            "Emlp1_to_attentionout", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
        )

        self.mlp_2 = RateCell(
            "mlp_2",
            n_units=dim,
            tau_m=tau_m,
            act_fx=act_fx,
            prior=("gaussian", 0.),
            integration_type='euler'
        )

        self.mlp_2_error = ErrorCell("mlp_2_error", n_units=dim)

        self.mlp2_zout = HebbianSynapse(
            "mlp2_zout",
            shape=(dim, dim),
            eta=eta,
            weight_init=dist.uniform(amin=wlb, amax=wub),
            bias_init=dist.constant(0.),
            w_bound=0.,
            optim_type=optim_type,
            sign_value=-1.,
            key=subkeys[4]
        )

        self.Ezout_mlp2 = StaticSynapse(
            "Ezout_mlp2", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
        )
        self.Emlp1_to_z_score = StaticSynapse(
            "Emlp1_to_z_score", shape=(dim, dim), weight_init=dist.uniform(amin=wlb, amax=wub), key=subkeys[5]
        )

    # Add this method
    def all_components(self):
        """Return a list of all internal cells and synapses for context registration"""
        return [
            self.mlp_1_error,
            self.mlp_1,
            self.mlp1_mlp2,
            self.Emlp2_mlp1,
            self.Emlp1_to_attentionout,
            self.mlp_2,
            self.mlp_2_error,
            self.mlp2_zout,
            self.Ezout_mlp2
        ]
