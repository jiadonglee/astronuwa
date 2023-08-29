import numpy as np
import pandas as pd
from tqdm import tqdm
from nuwa.stellarmodel import StellarEvolutionModel
from nuwa.nuwa import Nuwa_pop
import emcee

# Define your log-likelihood function
def log_likelihood(params, data):
    
    alpha, fb, gamma = params
    g_obs, bp_rp_obs  = data

    binary_model = Nuwa_pop(alpha, fb, gamma, parsec_model, 
                            n_star=2000, m_min=0.5, m_max=1.,
                            moh_min=-1, moh_max=0.5, 
                            backend='numpy')
    g_model, bp_rp_model = binary_model.mock_population()
    
    residuals = g_obs-g_model + bp_rp_obs-bp_rp_model
    
    chi_sq = np.sum(residuals**2)
    log_like = -0.5 * chi_sq
    return log_like

# Define your log-prior function
def log_prior(params):
    alpha, fb, gamma = params
    # Define your prior distribution for each parameter
    # Replace this with your own prior distribution
    if 0<alpha<5 and 0<fb<1 and 0<gamma<5:
        return 0.0
    return -np.inf


def log_normal_prior(params):
    alpha, fb, gamma = params
    # Define the bounds for each parameter
    alpha_min, alpha_max = 0, 5
    fb_min, fb_max = 0, 1
    gamma_min, gamma_max = 0, 5
    
    # Define the means and standard deviations for the Gaussian priors
    alpha_mean, alpha_std = 2.5, 0.5
    fb_mean, fb_std = 0.5, 0.2
    gamma_mean, gamma_std = 2.5, 0.5
    
    # Check if the parameters are within the bounds
    if alpha_min < alpha < alpha_max and fb_min < fb < fb_max and gamma_min < gamma < gamma_max:
        # Calculate the log probability of each parameter given the Gaussian prior
        alpha_log_prob = -0.5 * ((alpha - alpha_mean) / alpha_std)**2
        fb_log_prob = -0.5 * ((fb - fb_mean) / fb_std)**2
        gamma_log_prob = -0.5 * ((gamma - gamma_mean) / gamma_std)**2
        
        # Return the sum of the log probabilities as the prior
        return alpha_log_prob + fb_log_prob + gamma_log_prob
    
    return -np.inf

# Define your log-posterior function
def log_posterior(params, data):
    log_prior_val = log_prior(params)
    if np.isinf(log_prior_val):
        return log_prior_val
    return log_prior_val + log_likelihood(params, data)



if __name__ == "__main__":
    
    stellar_model_dir  = "/nfsdata/share/stellarmodel/"
    save_chain_dir = "/nfsdata/users/jdli_ny/nuwa/mcmc/"
    stellar_model_name = "PARSEC_logage9p5_MH_n1p5_0p6.csv"

    # settings
    alpha_true = 2.3
    fb_true = 0.5
    gamma_true = 1.1

    parsec_model = StellarEvolutionModel(stellar_model_dir+stellar_model_name)

    binary_pop = Nuwa_pop(alpha_true, fb_true, gamma_true, parsec_model, 
                      n_star=2000, m_min=0.5, m_max=1.,
                      moh_min=-1, moh_max=0.5, backend='numpy')

    g_mock, bp_rp_mock = binary_pop.mock_population()

    # add noise
    # Add Gaussian noise to g_mock
    g_mock_noisy = g_mock + np.random.normal(loc=0, scale=1e-3, size=g_mock.shape)

    # Add Gaussian noise to bp_rp_mock
    bp_rp_mock_noisy = bp_rp_mock + np.random.normal(loc=0, scale=1e-3, size=bp_rp_mock.shape)

    # mcmc
    # Initialize the number of walkers and dimensions
    ndim = 3  # Number of parameters (fb, gamma)
    nwalkers = 32  # Number of walkers

    # Initialize the walkers with random positions around a guess
    guess = [2, 0.5, 0.5]  # Initial guess for the parameters
    pos = guess + 1e-4 * np.random.randn(nwalkers, ndim)

    # Set up the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=((g_mock_noisy, bp_rp_mock_noisy),))


    # Run the MCMC sampling
    nsteps = 1000  # Number of MCMC steps
    burn_steps = 500  # Number of burn-in steps

    # Perform the burn-in phase
    pos, _, _ = sampler.run_mcmc(pos, burn_steps, progress=True)


    # Reset the sampler and run the production phase
    sampler.reset()
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Retrieve the samples
    samples = sampler.get_chain(discard=burn_steps, flat=True)

    np.save(save_chain_dir+"alpha_fb_gamma_sample.npy", samples)