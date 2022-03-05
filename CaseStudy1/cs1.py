# Import libraries
import numpy as np
import pandas as pd
from scipy.stats import poisson

# Parameters
N = 5 # dealer capacity
lambda_rate = 3 # arrival rate
c = 50000 # cost per unit
r = 90000 # revenue
K = 100000 # fixed cost
h = 5000 # inventory holding cost
b = 300000 # goodwill cost

S = 4 # order size
s_order = 1 # reorder point

x = S # initial condition

# %% Create matrix
probability_matrix = np.zeros(shape=(S+1, S+1), dtype=float)

# Filling out the matrix
for i in range(S+1):
    for j in range(S+1):
        if i <= s_order or i == S:
            if j > 0:
                probability_matrix[i, j] = poisson.pmf(S - j, lambda_rate)
            else:
                probability_matrix[i, j] = 1 - poisson.cdf(S - 1, lambda_rate)
        else:
            if j > i:
                probability_matrix[i, j] = 0
            elif j != 0:
                probability_matrix[i, j] = poisson.pmf(i - j, lambda_rate)
            else:
                probability_matrix[i, j] = 1 - poisson.cdf(i - 1, lambda_rate)

# %% Create the cost vector

profit_vector = np.zeros(shape=(S+1,), dtype=float)

for i in range(S+1):
    if i <= s_order:
        profit_vector[i] = -h * i + r * (np.sum([poisson.pmf(k, lambda_rate) * k for k in range(0, S)]) + S * (
                    1 - poisson.cdf(S - 1, lambda_rate))) - b * (1 - poisson.cdf(S, lambda_rate)) - K - c * (S - i)
    else:
        profit_vector[i] = -h * i + r * (np.sum([poisson.pmf(k, lambda_rate) * k for k in range(0, i)]) + i * (
                    1 - poisson.cdf(i - 1, lambda_rate))) - b * (1 - poisson.cdf(i, lambda_rate))


# %% Create excel

pd.DataFrame(np.vstack([probability_matrix, profit_vector]))