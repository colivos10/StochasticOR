import numpy as np

def log_normal_variates(mu, sigma, replicates):

    x_log = []
    for i in range(replicates):
        u1 = np.random.uniform()
        u2 = np.random.uniform()

        #box miller
        z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

        # log normal
        x = np.exp(mu + sigma * np.random.choice([z1, z2]))
        x_log.append(x)

    return np.mean(x_log), np.var(x_log)