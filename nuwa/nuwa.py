import numpy as np
import copy
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
from nuwa.stellarmodel import StellarEvolutionModel
import numpyro
from numpyro import distributions as dist, infer
import jax.random as random
import jax
import jax.numpy as jnp
from jax import jit, vmap
from tqdm import tqdm
jax.config.update('jax_platform_name', 'cpu')



class Nuwa_pop_mono:
    def __init__(self, alpha, fb, gamma, stellar_evolution_model, m_min=0.1, m_max=2, n_star=5000, seed=42):
        self.alpha = alpha
        self.fb = fb
        self.gamma = gamma
        self.stellar_evolution_model = stellar_evolution_model
        self.alpha = alpha
        self.m_min  = m_min
        self.m_max  = m_max
        self.n_star = n_star
        self.random_state = np.random.default_rng(seed=seed)
    
    def mock_population(self):
        self.masses1 = self._generate_masses()
        
        # 1st star
        self.mass_ratios = self._generate_mass_ratios()
        self.G1, self.Bp1, self.Rp1 = self.stellar_evolution_model.compute_gaia_bands(self.masses1)
        
        # 2nd star
        _is_binary = self._generate_binary_fraction()
        
        self.masses2 = self.mass_ratios * self.masses1
        self.masses2[self.masses2<=self.m_min] = self.m_min
        self.masses2[self.masses2>=self.m_max] = self.m_max
        self.G2, self.Bp2, self.Rp2 = self.stellar_evolution_model.compute_gaia_bands(self.masses2)
        
        # combine total magnitudes 
        self.G_tot  = self._combine_mags(self.G1, self.G2)
        self.Bp_tot = self._combine_mags(self.Bp1, self.Bp2)
        self.Rp_tot = self._combine_mags(self.Rp1, self.Rp2)
        
        self._is_binary_ind = _is_binary==1
        self.mags = copy.copy(self.G1)
        self.mags[self._is_binary_ind] = self.G_tot[self._is_binary_ind]
        
        self.colors = self.Bp1 - self.Rp1
        self.colors[self._is_binary_ind] = self.Bp_tot[self._is_binary_ind] - self.Rp_tot[self._is_binary_ind]
        
        return self.mags, self.colors
    
    def _combine_mags(self, mag1, mag2):
        flux1 = 10 ** (-0.4 * mag1)  # Convert magnitude to flux in linear space
        flux2 = 10 ** (-0.4 * mag2)
        combined_flux = flux1 + flux2  # Add the fluxes
        combined_mag = -2.5 * np.log10(combined_flux)  # Convert combined flux back to magnitude
        return combined_mag
    
    def _generate_masses(self):
        # Generate masses from the initial mass function (IMF)
        cdf = self.random_state.uniform(size=self.n_star)
        mass_scale = cdf * (self.m_max ** (1 - self.alpha) - self.m_min ** (1 - self.alpha)) + self.m_min ** (1 - self.alpha)
        return mass_scale ** (1 / (1 - self.alpha))
    
    def _generate_mass_ratios(self):
        # Generate mass ratios from the mass-ratio distribution
        cdf = self.random_state.uniform(size=self.n_star)
        mass_scale = cdf * (1 ** (1 + self.gamma) - 0.1 ** (1 + self.gamma)) + 0.1 ** (1 + self.gamma)
        return mass_scale ** (1 / (1 + self.gamma))
    
    def _generate_binary_fraction(self):
        binary_class = self.random_state.binomial(1, self.fb, self.n_star)
        return binary_class



class Nuwa_pop:

    def __init__(self, alpha: float, 
                 fb: float, 
                 gamma: float,
                 stellar_evolution_model,
                 m_min: float = 0.5,
                 m_max: float = 1.0,
                 moh_min: float = -1,  
                 moh_max: float = 0.3,
                 n_star: int = 2000,
                 seed: int = 42,
                 backend: str = 'jax'):
        
        self.alpha = alpha
        self.fb = fb
        self.gamma = gamma

        self.imf_norm = (m_max**(1-alpha) - m_min**(1-alpha)) + m_min**(1-alpha)

        self.q_norm = (1.**(1-gamma) - 0.1**(1-gamma)) + 0.1**(1-gamma)

        self.m_min  = m_min
        self.m_max  = m_max
        self.moh_min = moh_min
        self.moh_max = moh_max
        self.n_star  = n_star
        self.backend = backend
        self.random_state = self._initialize_random(seed, backend)
        self.stellar_evolution_model = stellar_evolution_model

    def _initialize_random(self, seed, backend):

        if backend=='numpy':
            return np.random.default_rng()
        elif backend=='jax':
            return random.PRNGKey(seed)
    

    def mock_population(self):

        self.masses1 = self._generate_masses(self.n_star)
        
        # Assume a univsersal IMF with [M/H]
        if self.backend=="jax":

            self.moh = random.uniform(self.random_state, minval=self.moh_min, maxval=self.moh_max, shape=(self.n_star,))

        elif self.backend=='numpy':
            self.moh = np.random.uniform(self.moh_min, self.moh_max, size=self.n_star)
        
        # 1st star
        self.mass_ratios = self._generate_mass_ratios(self.n_star)
        self.G1, self.Bp1, self.Rp1 = self.stellar_evolution_model.compute_gaia_bands(self.masses1, self.moh)

        self.colors = copy.copy(self.Bp1 - self.Rp1)
        self.mags   = copy.copy(self.G1)
        
        # 2nd star
        _is_binary = self._generate_binary_fraction(self.n_star)
        
        self.masses2 = self.mass_ratios * self.masses1

        self.masses2 = np.where(self.masses2<=0.1, 0.1, self.masses2)
        self.masses2 = np.where(self.masses2>=self.m_max, self.m_max, self.masses2)

        self.G2, self.Bp2, self.Rp2 = self.stellar_evolution_model.compute_gaia_bands(self.masses2, self.moh)
        
        # combine total magnitudes 
        self.G_tot  = self._combine_mags(self.G1, self.G2)
        self.Bp_tot = self._combine_mags(self.Bp1, self.Bp2)
        self.Rp_tot = self._combine_mags(self.Rp1, self.Rp2)

        self._is_binary_ind = _is_binary==1

        if self.backend=='numpy':

            self.mags[self._is_binary_ind] = self.G_tot[self._is_binary_ind]
            self.colors[self._is_binary_ind] = self.Bp_tot[self._is_binary_ind] - self.Rp_tot[self._is_binary_ind]

        elif self.backend=='jax':

            mag_pop = self.G_tot[self._is_binary_ind]
            colors_pop = self.Bp_tot[self._is_binary_ind] - self.Rp_tot[self._is_binary_ind]

            self.mags   = jnp.concatenate([mag_pop, self.mags[~self._is_binary_ind]], axis=0)
            self.colors = jnp.concatenate([colors_pop, self.colors[~self._is_binary_ind]], axis=0)

        return self.mags, self.colors
    

    def _combine_mags(self, mag1, mag2):
        flux1 = 10** (-0.4 * mag1)  # Convert magnitude to flux in linear space
        flux2 = 10** (-0.4 * mag2)
        combined_flux = flux1 + flux2  # Add the fluxes

        if self.backend=='numpy':
            return -2.5 * np.log10(combined_flux)  # Convert combined flux back to magnitude
        elif self.backend=='jax':
            return -2.5 * jnp.log10(combined_flux)
    

    def _generate_masses(self, num_star):

        if self.backend=='numpy':
            cdf = np.random.uniform(size=num_star)
            mass_scale = cdf*(self.m_max ** (1-self.alpha) - self.m_min**(1-self.alpha)) + self.m_min**(1-self.alpha)

        elif self.backend=='jax':
            key, subkey = random.split(self.random_state)
            cdf = random.uniform(key, shape=(num_star,))
            mass_scale = cdf*(self.m_max ** (1-self.alpha) - self.m_min**(1-self.alpha)) + self.m_min**(1-self.alpha)

        return mass_scale ** (1 / (1 - self.alpha))
    

    def _generate_mass_ratios(self, num_star):

        if self.backend=='numpy':
            cdf = np.random.uniform(size=num_star)
            mass_scale = cdf*(1 ** (1+self.gamma) - 0.1**(1+self.gamma)) + 0.1**(1+self.gamma)

        elif self.backend=='jax':
            _, subkey = random.split(self.random_state)
            cdf = random.uniform(subkey, shape=(num_star,))
            mass_scale = cdf*(1 ** (1+self.gamma) - 0.1**(1+self.gamma)) + 0.1**(1+self.gamma)
        
        return mass_scale ** (1 / (1+self.gamma))
    

    def _generate_binary_fraction(self, num_star):

        if self.backend=='numpy':
            binary_class = np.random.binomial(1, self.fb, num_star)

        elif self.backend=='jax':
            key, subkey  = random.split(self.random_state)
            binary_class = random.bernoulli(subkey, p=self.fb, shape=(num_star,))
        return binary_class
    
    def fit(self, mags=None, colors=None):
        
        """under construction"""
        numpyro.set_rng_seed(42)

        # Prior distributions
        alpha = dist.Uniform(0, 3)
        fb    = dist.Uniform(0, 1) 
        gamma = dist.Uniform(0, 3)

        # Sample parameters
        params = numpyro.sample([alpha, fb, gamma], strategy='slice')  

        # Generate population
        self = Nuwa_pop(params[0], params[1], params[2])

        simulated_mags, simulated_colors = self.mock_population()

        # Likelihood
        numpyro.sample('obs', dist.Normal(simulated_mags, 0.02), obs=mags)
        numpyro.sample('obs', dist.Normal(simulated_colors, 0.02), obs=colors)




if __name__ == "__main__":
    
    stellar_model_dir  = "/nfsdata/share/stellarmodel/"
    save_chain_dir = "/nfsdata/users/jdli_ny/nuwa/mcmc/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"

    # settings
    # alpha_true = 2.3
    # fb_true = 0.5
    # gamma_true = 1.1

    parsec_model = StellarEvolutionModel(
        stellar_model_dir+stellar_model_name, backend="jax"
        )
    # population_mocker = BinaryPopulationMocker_numpy(alpha_true, fb_true, gamma_true, parsec_model)

    # g_mock, bp_rp_mock = population_mocker.mock_population()
    # print(g_mock, bp_rp_mock)

    n_train = 200
    n_star  = 2000
    # Define the parameters for the uniform distributions
    alpha_min, alpha_max = 1.3,  3.
    fb_min,    fb_max    = 0.01, 0.99
    gamma_min, gamma_max = 1.3,  4.

    """uniform parameter space"""
    # # Set the random seed for reproducibility (optional)
    np.random.seed(42)

    # Generate random values for fb and gamma
    # alpha_true = 2.3
    alpha_values = np.random.uniform(alpha_min, alpha_max, size=n_train)
    fb_values    = np.random.uniform(fb_min,    fb_max,    size=n_train)
    gamma_values = np.random.uniform(gamma_min, gamma_max, size=n_train)

    X = []
    Y = []


    for alpha, fb, gamma in tqdm(zip(alpha_values, fb_values, gamma_values)):
        
        # print(r"alpha=%.2f, f_b=%.2f, gamma=%.2f"%(alpha, fb, gamma))

        pop = Nuwa_pop(
            alpha, fb, gamma, parsec_model, 
            n_star=n_star, m_min=0.5, m_max=1.,
            moh_min=-0.5, moh_max=0, 
            backend='jax'
            )
        g_mock, bp_rp_mock = pop.mock_population()
        
        X.extend([[bp_rp, gmag] for bp_rp, gmag in zip(bp_rp_mock, g_mock)])
        Y.extend([[alpha, fb, gamma]])


    X = np.array(X)
    X = X.reshape(n_train, n_star, 2)
    Y = np.array(Y)

    print(X.shape, Y.shape)


    data_dir = "/nfsdata/users/jdli_ny/wlkernel/mock/"
    fname = data_dir+f'binary_train_moh_m0p5_0_abg_{n_train}tr_{n_star}cmd.npz'
    np.savez(fname, X=X, Y=Y)