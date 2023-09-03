import numpy as np
import copy
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit, random
import objax
from neural import NeuralNetwork
import objax
# os.environ["XLA_GPU_DEVICE_ORDINAL"] = "gpu:2"
# jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
# num_devices = jax.desvice_count()
# print(num_devices)
# from jax import random, vmap



class StellarEvolutionModel:
    
    def __init__(self, data_file, backend="numpy"):
        self.data = pd.read_csv(data_file)
        self.data = self.data.sort_values('Mini')
        self.mass_range = (self.data['Mini'].min(), self.data['Mini'].max())
        self.backend = backend
        self._create_interpolators()
    
    def _create_interpolators(self):

        if self.backend == "numpy":
            """if numpy, then simply scipy.LinearNDInterpolator"""

            self._gmag_interpolator     = LinearNDInterpolator(
                list(zip(self.data['Mini'], self.data['MH'])), self.data['Gmag']
            )
            self._g_bp_mag_interpolator = LinearNDInterpolator(
                list(zip(self.data['Mini'], self.data['MH'])), self.data['G_BPmag']
            )
            self._g_rp_mag_interpolator = LinearNDInterpolator(
                list(zip(self.data['Mini'], self.data['MH'])), self.data['G_RPmag']
            )

        elif self.backend == "jax":
            
            stellar_model_dir  = "/nfsdata/share/stellarmodel/"

            model_G = NeuralNetwork(2, 3, 1)
            model_Bp = NeuralNetwork(2, 5, 1)
            model_Rp = NeuralNetwork(2, 5, 1)

            with open(stellar_model_dir+'mass2G.npz', 'rb') as f:
                objax.io.load_var_collection(f, model_G.vars())

            with open(stellar_model_dir+'mass2Bp.npz', 'rb') as f:
                objax.io.load_var_collection(f, model_Bp.vars())

            with open(stellar_model_dir+'mass2Rp.npz', 'rb') as f:
                objax.io.load_var_collection(f, model_Rp.vars())

            self._gmag_interpolator = model_G
            self._g_bp_mag_interpolator = model_Bp
            self._g_rp_mag_interpolator = model_Rp


    def compute_magnitudes(self, masses, mohs):

        if self.backend == "numpy":
            magnitudes = self._gmag_interpolator(masses, mohs)

        elif self.backend == "jax":
            magnitudes = self._gmag_interpolator(jnp.vstack((masses, mohs)))

        return magnitudes
    
    def compute_colors(self, masses, mohs):

        if self.backend == "numpy":
            g_bp_mag = self._g_bp_mag_interpolator(masses, mohs)
            g_rp_mag = self._g_rp_mag_interpolator(masses, mohs)
        
        elif self.backend == "jax":
            g_bp_mag = self._g_bp_mag_interpolator(jnp.vstack(masses, mohs))
            g_rp_mag = self._g_rp_mag_interpolator(jnp.vstack(masses, mohs))
        
        colors = g_bp_mag - g_rp_mag
        return colors
    
    def compute_gaia_bands(self, masses, mohs):

        if self.backend == "numpy":

            gmag = self._gmag_interpolator(masses, mohs)
            g_bp_mag = self._g_bp_mag_interpolator(masses, mohs)
            g_rp_mag = self._g_rp_mag_interpolator(masses, mohs)
        
        elif self.backend == "jax":

            vars = jnp.vstack((masses, mohs))

            gmag = self._gmag_interpolator(vars)
            g_bp_mag = self._g_bp_mag_interpolator(vars)
            g_rp_mag = self._g_rp_mag_interpolator(vars)

        return gmag, g_bp_mag, g_rp_mag


# @jit(static_argnums=(0,))
def init_network_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    return [random.normal(k, (m, n)) / jnp.sqrt(n)
            for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)]

@jit
def forward(params, inputs):
    activations = inputs
    for w in params[:-1]:
        activations = jnp.tanh(jnp.dot(activations, w))
    return jnp.dot(activations, params[-1])

@jit
def loss(params, inputs, targets):
    preds = forward(params, inputs)
    return jnp.mean((preds - targets)**2)


@jit
def update(params, inputs, targets, learning_rate):
    grads = grad(loss)(params, inputs, targets)
    return [(w - learning_rate * dw) for w, dw in zip(params, grads)]

@jit
def _test_fc(inputs, targets):
    
    key = random.PRNGKey(0)
    layer_sizes = [2, 10, 1]  # Input size, hidden size, output size
    params = init_network_params(layer_sizes, key)

    learning_rate = 0.1
    num_epochs = 100

    for epoch in range(num_epochs):
        params = update(params, inputs, targets, learning_rate)

        if epoch % 10 == 0:
            loss_value = loss(params, inputs, targets)
            print(f"Epoch {epoch}, Loss: {loss_value}")



if __name__ == "__main__":
    
    stellar_model_dir  = "/nfsdata/share/stellarmodel/"
    save_chain_dir     = "/nfsdata/users/jdli_ny/nuwa/mcmc/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"

    parsec_model = StellarEvolutionModel(
        stellar_model_dir+stellar_model_name, 
        backend="numpy"
        )
    
    out = parsec_model.compute_gaia_bands(
        jnp.array([0.5, 0.3]), jnp.array([-0.2, -0.3]))
    
    print(out)
    # print(jax.devices()[0])
    # key = random.PRNGKey(0)

    # device = 'cpu'
    # key = jax.devices()[5]  # Get the first GPU device
    # jax.config.update('jax_platform_name', key)
    # 
    # for i, device in enumerate(devices):
    #     print(f"Device {i}: {device}")
    # print(f'Number of GPUs {jax.device_count()}')
    
    # x = objax.random.normal(shape=(100, 4))
    # m = objax.nn.Linear(nin=4, nout=5)
    # print('Matrix product shape', m(x).shape)  # (100, 5
    