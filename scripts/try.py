import numpy as np 
import matplotlib.pyplot as plt

def simulate_stochastic_volatility(n_steps, mu, phi, sigma, h0):
    """
    Simulates a time series from a simple stochastic volatility model.

    Args:
        n_steps (int): The number of time steps to simulate.
        mu (float): The mean log volatility.
        phi (float): The persistence of volatility (between -1 and 1).
        sigma (float): The standard deviation of the log-volatility shock.
        h0 (float): The initial log volatility.

    Returns:
        tuple: (returns, log_volatility)
    """
    returns = np.zeros(n_steps)
    log_volatility = np.zeros(n_steps)
    log_volatility[0] = h0
    
    # Generate random shocks
    epsilons = np.random.normal(0, 1, n_steps)
    ws = np.random.normal(0, 1, n_steps)

    for t in range(1, n_steps):
        # State equation (log volatility)
        log_volatility[t] = mu + phi * (log_volatility[t-1] - mu) + sigma * ws[t]
        
        # Observation equation (returns)
        returns[t] = np.exp(log_volatility[t] / 2) * epsilons[t]
        
    return returns, log_volatility



D = simulate_stochastic_volatility(200, 0.0, 1.0, 0.98, 0.0)

plt.plot(D[0])
plt.plot(D[1])
plt.show() 



