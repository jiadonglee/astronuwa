import numpy as np
import copy
from scipy.interpolate import LinearNDInterpolator
import pandas as pd
from model import StellarEvolutionModel
import numpyro
from numpyro import distributions as dist, infer
import jax.random as random
import jax
import jax.numpy as jnp
from jax import jit, vmap
jax.config.update('jax_platform_name', 'cpu')



class BinaryPopulationMocker:
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



class BinaryPopulationMocker_mdf:

    def __init__(self, alpha, fb, gamma, stellar_evolution_model,
                 m_min=0.5, m_max=1., moh_min=-1, moh_max=0.3,
                 n_star=2000, seed=42):
        
        self.alpha = alpha
        self.fb = fb
        self.gamma = gamma
        self.stellar_evolution_model = stellar_evolution_model
        self.alpha  = alpha
        self.m_min  = m_min
        self.m_max  = m_max
        self.moh_min = moh_min
        self.moh_max = moh_max
        self.n_star  = n_star
        # self.random_state = np.random.default_rng(seed=seed)
        self.random_state = random.PRNGKey(seed)
    
    def mock_population(self):
        self.masses1 = self._generate_masses(self.n_star)
        
        # Assume a univsersal IMF with [M/H]
        self.moh = random.uniform(self.random_state, minval=self.moh_min, maxval=self.moh_max)
        
        # 1st star
        self.mass_ratios = self._generate_mass_ratios(self.n_star)
        self.G1, self.Bp1, self.Rp1 = self.stellar_evolution_model.compute_gaia_bands(self.masses1, self.moh)
        
        # 2nd star
        _is_binary = self._generate_binary_fraction(self.n_star)
        
        self.masses2 = self.mass_ratios * self.masses1
        self.masses2 = jnp.where(self.masses2 <= self.m_min, self.m_min, self.masses2)
        self.masses2 = jnp.where(self.masses2 >= self.m_max, self.m_max, self.masses2)
        self.G2, self.Bp2, self.Rp2 = self.stellar_evolution_model.compute_gaia_bands(self.masses2, self.moh)
        
        # combine total magnitudes 
        self.G_tot  = self._combine_mags(self.G1, self.G2)
        self.Bp_tot = self._combine_mags(self.Bp1, self.Bp2)
        self.Rp_tot = self._combine_mags(self.Rp1, self.Rp2)

        # self._is_binary_ind = _is_binary==1
        self._is_binary_ind = _is_binary==1

        self.mags = copy.copy(self.G1)
        self.mags.at[self._is_binary_ind].set(self.G_tot[self._is_binary_ind])
        
        self.colors = self.Bp1 - self.Rp1
        self.colors.at[self._is_binary_ind].set(self.Bp_tot[self._is_binary_ind] - self.Rp_tot[self._is_binary_ind])
        return self.mags, self.colors
    

    def _combine_mags(self, mag1, mag2):
        flux1 = jnp.power(10, -0.4 * mag1)  # Convert magnitude to flux in linear space
        flux2 = jnp.power(10, -0.4 * mag2)
        combined_flux = flux1 + flux2  # Add the fluxes
        combined_mag = -2.5 * jnp.log10(combined_flux)  # Convert combined flux back to magnitude
        return combined_mag
    

    def _generate_masses(self, num_star):
        # Generate masses from the initial mass function (IMF)
        # cdf = self.random_state.uniform(size=num_star)
        # cdf = numpyro.sample('cdf_m', dist.Uniform(0., 1.))
        cdf = random.uniform(self.random_state, shape=(num_star,))
        mass_scale = cdf * (self.m_max ** (1 - self.alpha) - self.m_min ** (1 - self.alpha)) + self.m_min ** (1 - self.alpha)
        return jnp.power(mass_scale, (1 / (1 - self.alpha)))
    

    def _generate_mass_ratios(self, num_star):
        # Generate mass ratios from the mass-ratio distribution
        # cdf = self.random_state.uniform(size=num_star)
        # cdf = numpyro.sample('cdf_q', dist.Uniform(0., 1.))
        cdf = random.uniform(self.random_state, shape=(num_star,))
        mass_scale = cdf * (1 ** (1 + self.gamma) - 0.1 ** (1 + self.gamma)) + 0.1 ** (1 + self.gamma)
        return jnp.power(mass_scale, (1 / (1 + self.gamma)))
    

    def _generate_binary_fraction(self, num_star):
        # binary_class = self.random_state.binomial(1, self.fb, num_star)
        binary_class = random.bernoulli(self.random_state, p=self.fb, shape=(num_star,))
        # binary_class = numpyro.sample("fb", dist.Bernoulli(probs=self.fb))
        return binary_class


class BinaryPopulationMocker_numpy:
    def __init__(self, alpha, fb, gamma, stellar_evolution_model, m_min=0.1, m_max=1., n_star=5000, seed=42):
        self.alpha = alpha
        self.fb = fb
        self.gamma = gamma
        self.stellar_evolution_model = stellar_evolution_model
        self.alpha = alpha
        self.m_min  = m_min
        self.m_max  = m_max
        self.moh_min = -1
        self.moh_max = 0.3
        self.n_star = n_star
        self.random_state = np.random.default_rng(seed=seed)
    
    def mock_population(self):
        self.masses1 = self._generate_masses()
        
        # Assume a univsersal IMF with [M/H]
        self.moh = self.random_state.uniform(self.moh_min, self.moh_max, self.n_star)
        
        # 1st star
        self.mass_ratios = self._generate_mass_ratios()
        self.G1, self.Bp1, self.Rp1 = self.stellar_evolution_model.compute_gaia_bands(self.masses1, self.moh)
        
        # 2nd star
        _is_binary = self._generate_binary_fraction()
        
        self.masses2 = self.mass_ratios * self.masses1
        self.masses2[self.masses2<=self.m_min] = self.m_min
        self.masses2[self.masses2>=self.m_max] = self.m_max
        self.G2, self.Bp2, self.Rp2 = self.stellar_evolution_model.compute_gaia_bands(self.masses2, self.moh)
        
        # combine total magnitudes 
        self.G_tot  = self._combine_mags(self.G1, self.G2)
        self.Bp_tot = self._combine_mags(self.Bp1, self.Bp2)
        self.Rp_tot = self._combine_mags(self.Rp1, self.Rp2)
        
        self._is_binary_ind = _is_binary==1

        self.mags = copy.copy(self.G1)
        self.mags.at[self._is_binary_ind].set(self.G_tot[self._is_binary_ind])
        
        self.colors = self.Bp1 - self.Rp1
        self.colors.at[self._is_binary_ind].set(self.Bp_tot[self._is_binary_ind] - self.Rp_tot[self._is_binary_ind])
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




if __name__ == "__main__":
    
    stellar_model_dir  = "/nfsdata/share/stellarmodel/"
    save_chain_dir = "/nfsdata/users/jdli_ny/nuwa/mcmc/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"

    # settings
    # alpha_true = 2.3
    # fb_true = 0.5
    # gamma_true = 1.1

    parsec_model = StellarEvolutionModel(stellar_model_dir+stellar_model_name)
    # population_mocker = BinaryPopulationMocker_numpy(alpha_true, fb_true, gamma_true, parsec_model)

    # g_mock, bp_rp_mock = population_mocker.mock_population()
    # print(g_mock, bp_rp_mock)

    n_train = 500
    n_star  = 5000

    # Define the parameters for the uniform distributions
    alpha_min, alpha_max = 1.1, 4.
    fb_min, fb_max = 0.01, 0.99
    gamma_min, gamma_max = -0.9, 4.

    # Set the random seed for reproducibility (optional)
    np.random.seed(42)

    # Generate random values for fb and gamma
    # alpha_true = 2.3
    alpha_values = np.random.uniform(alpha_min, alpha_max, size=n_train)
    fb_values    = np.random.uniform(fb_min,    fb_max,    size=n_train)
    gamma_values = np.random.uniform(gamma_min, gamma_max, size=n_train)

    X = []
    Y = []


    for alpha, fb, gamma in zip(alpha_values, fb_values, gamma_values):
        
        print(r"alpha=%.2f, f_b=%.2f, gamma=%.2f"%(alpha, fb, gamma))

        population_mocker = BinaryPopulationMocker_numpy(
            alpha, fb, gamma, parsec_model, 
            n_star=n_star
            )
        g_mock, bp_rp_mock = population_mocker.mock_population()
        
        X.extend([[bp_rp, gmag] for bp_rp, gmag in zip(bp_rp_mock, g_mock)])
        Y.extend([[alpha, fb, gamma]])


    X = np.array(X)
    X = X.reshape(n_train, n_star, 2)
    Y = np.array(Y)

    print(X.shape, Y.shape)

    np.savez('/nfsdata/users/jdli_ny/wlkernel/mock/binary_train_flatZ_abg.npz', X=X, Y=Y)