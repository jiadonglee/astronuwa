import numpy as np
import copy
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap

# from jax import random, vmap
jax.config.update('jax_platform_name', 'cpu')


class StellarEvolutionModel:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.data = self.data.sort_values('Mini')
        self.mass_range = (self.data['Mini'].min(), self.data['Mini'].max())
        self._create_interpolators()
    
    def _create_interpolators(self):
        self._gmag_interpolator     = LinearNDInterpolator(
            list(zip(self.data['Mini'], self.data['MH'])), self.data['Gmag']
        )
        self._g_bp_mag_interpolator = LinearNDInterpolator(
            list(zip(self.data['Mini'], self.data['MH'])), self.data['G_BPmag']
        )
        self._g_rp_mag_interpolator = LinearNDInterpolator(
            list(zip(self.data['Mini'], self.data['MH'])), self.data['G_RPmag']
        )
    
    def compute_magnitudes(self, masses, mohs):
        magnitudes = vmap(lambda x, y: self._gmag_interpolator(masses, mohs))(masses, mohs)
        # magnitudes = self._gmag_interpolator(masses, mohs)
        return jnp.array(magnitudes)
    
    def compute_colors(self, masses, mohs):
        g_bp_mag = self._g_bp_mag_interpolator(masses, mohs)
        g_rp_mag = self._g_rp_mag_interpolator(masses, mohs)
        # colors = g_bp_mag - g_rp_mag
        # g_bp_mag = vmap(lambda x, y: self._g_bp_mag_interpolator(masses, mohs))(masses, mohs)
        # g_rp_mag = vmap(lambda x, y: self._g_rp_mag_interpolator(masses, mohs))(masses, mohs)
        colors = g_bp_mag - g_rp_mag
        return jnp.array(colors)
    
    def compute_gaia_bands(self, masses, mohs):
        gmag = self._gmag_interpolator(masses, mohs)
        g_bp_mag = self._g_bp_mag_interpolator(masses, mohs)
        g_rp_mag = self._g_rp_mag_interpolator(masses, mohs)
        # gmag     = vmap(lambda x, y: self._gmag_interpolator(masses, mohs))(masses, mohs)
        # g_bp_mag = vmap(lambda x, y: self._g_bp_mag_interpolator(masses, mohs))(masses, mohs)
        # g_rp_mag = vmap(lambda x, y: self._g_rp_mag_interpolator(masses, mohs))(masses, mohs)
        return jnp.array(gmag), jnp.array(g_bp_mag), jnp.array(g_rp_mag)
    

if __name__ == "__main__":
    
    stellar_model_dir  = "/nfsdata/share/stellarmodel/"
    save_chain_dir = "/nfsdata/users/jdli_ny/nuwa/mcmc/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"

    parsec_model = StellarEvolutionModel(stellar_model_dir+stellar_model_name)
    # print(parsec_model.compute_colors(1, 0))
    # print(type(parsec_model.compute_colors(1, 0)))
