import numpy as np
from sympy.abc import t


def log_normal_variates(mu, sigma, replicates):
    u1 = np.random.uniform()
    u2 = np.random.uniform()

    # box miller
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    # log normal
    x = np.exp(mu + sigma * np.random.choice([z1, z2]))

    return x

def non_homogenous_pp(replicates, formula, lambda_star):
    times_vector = []
    time = 0
    uniform_variate = np.random.uniform()  # Uniform(0,1)
    time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star  # x_n
    time = time + time_arrival_variate  # t = t + x_n
    while time <= 24:
        N = 0

        uniform_variate_hat = np.random.uniform()  # Uhat Uniform(0,1)
        lambda_t_return = formula.subs(t, time)

        if uniform_variate_hat <= lambda_t_return / lambda_star:
            N = N + 1
            times_vector.append(time)

        uniform_variate = np.random.uniform()  # Uniform(0,1)
        time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star  # x_n
        time = time + time_arrival_variate  # t = t + x_n

    return len(times_vector)


def non_homogenous_compound_pp(replicates, formula, lambda_star):
    times_vector = []
    email_size = []
    time = 0
    uniform_variate = np.random.uniform()  # Uniform(0,1)
    time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star  # x_n
    time = time + time_arrival_variate  # t = t + x_n
    N = 0
    while time <= 24:

        uniform_variate_hat = np.random.uniform()  # Uhat Uniform(0,1)
        lambda_t_return = formula.subs(t, time)

        if uniform_variate_hat <= lambda_t_return / lambda_star:
            N = N + 1
            times_vector.append(time)
            email_size.append(log_normal_variates(4, 1, 1))

        uniform_variate = np.random.uniform()  # Uniform(0,1)
        time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star  # x_n
        time = time + time_arrival_variate  # t = t + x_n

    return times_vector, email_size
