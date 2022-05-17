# %% Import libraries
import numpy as np
import pandas as pd
import scipy
# %% Parameters
c = 20
l = 200
mu = 15
eta = 5
nu = 50
gamma = 0.9
k = 0.5

# Variables
n = 11
m = 5

# %%  Generator matrix Q
def gen_matrix(c, l, mu, eta, nu, gamma, k, n, m):
    gen_matrix = np.zeros(shape=(c+1, c+1), dtype=float)
    gen_matrix[0, 0] = -(l + nu * k)
    gen_matrix[0, 1] = (l + nu * k)

    for i in range(1, m+1):
        gen_matrix[i, i - 1] = i * mu
        gen_matrix[i, i + 1] = l + nu * k
        gen_matrix[i, i] = -(gen_matrix[i, i - 1] + gen_matrix[i, i + 1])

    for i in range(m+1, n):
        gen_matrix[i, i - 1] = i * mu
        gen_matrix[i, i + 1] = l
        gen_matrix[i, i] = -(gen_matrix[i, i - 1] + gen_matrix[i, i + 1])

    for i in range(n, c):
        gen_matrix[i, i - 1] = n * mu + (i - n) * eta
        gen_matrix[i, i + 1] = l * gamma
        gen_matrix[i, i] = -(gen_matrix[i, i - 1] + gen_matrix[i, i + 1])

    gen_matrix[c, c-1] = n * mu + (c - n) * eta
    gen_matrix[c, c] = - (n * mu + (c - n) * eta)

    return gen_matrix

# %% Question 3 Uniformization and P function

def probability_nowait(gen_matrix, servers):
    global_lambda = max(-np.diag(gen_matrix))
    matrix_p = np.identity(c+1) + gen_matrix/global_lambda

    t = 1/6
    gen_matrix_t = gen_matrix * t

    matrix_p_t = 0
    for i in range(150):
        matrix_p_t = matrix_p_t + (np.exp(-global_lambda * t) * (global_lambda * t)**i/np.math.factorial(i)) * np.linalg.matrix_power(matrix_p, i)

    return sum(matrix_p_t[0][0:servers])

# %% Question 3
no_wait_saver = []
for thold in range(5, 9):
    no_wait_it = []
    for servers in range(10, 21):
        gen_matrix_it = gen_matrix(c, l, mu, eta, nu, gamma, k, servers, thold)
        no_wait_it.append(probability_nowait(gen_matrix_it, servers))
    no_wait_saver.append(no_wait_it)
no_wait_saver = np.array(no_wait_saver)

# %% Question 4 function

def compute_long_run_pbb(generation_matrix):
    gen_matrix_tilde = generation_matrix
    gen_matrix_tilde = np.column_stack([gen_matrix_tilde[:, 0:-1], np.ones(shape=(c + 1,))])

    vector_en = np.zeros((c+1))
    vector_en[-1] = 1

    gen_matrix_tilde_inv = np.linalg.inv(gen_matrix_tilde)

    long_run_pbb = np.matmul(vector_en, gen_matrix_tilde_inv)

    return long_run_pbb

# %% Question 4.a

rate_rejected_saver = []
for thold in range(5, 9):
    rate_rejected_it = []
    for servers in range(10, 21):
        gen_matrix_it = gen_matrix(c, l, mu, eta, nu, gamma, k, servers, thold)
        busy_signal_rate = l * compute_long_run_pbb(gen_matrix_it)[-1]
        rate_rejected_it.append(busy_signal_rate)
    rate_rejected_saver.append(rate_rejected_it)

rate_rejected_saver = np.array(rate_rejected_saver)

# %% Question 4.b

rate_hangup_saver = []
for thold in range(5, 9):
    rate_hangup_it = []
    for servers in range(10, 21):
        gen_matrix_it = gen_matrix(c, l, mu, eta, nu, gamma, k, servers, thold)
        busy_signal_rate = l * (1 - gamma) * sum(compute_long_run_pbb(gen_matrix_it)[servers:c])
        rate_hangup_it.append(busy_signal_rate)
    rate_hangup_saver.append(rate_hangup_it)

rate_hangup_saver = np.array(rate_hangup_saver)

# %% Question 5

revenue = 16
cost = 10

profit_saver = []
for thold in range(5, 9):
    profit_it = []
    for servers in range(10, 21):
        gen_matrix_it = gen_matrix(c, l, mu, eta, nu, gamma, k, servers, thold)
        long_run_pbb_vector = compute_long_run_pbb(gen_matrix_it)
        profit_vector = np.array([revenue * min(i, servers) - cost * servers for i in range(c+1)])
        expected_profit = np.matmul(long_run_pbb_vector, profit_vector)

        #expected_profit = revenue * (sum(i * long_run_pbb_vector[i] for i in range(1, servers)) +
        #                              servers * sum(long_run_pbb_vector[i] for i in range(servers, c+1))) - cost * servers
        profit_it.append(expected_profit)
    profit_saver.append(profit_it)

profit_saver = np.array(profit_saver)

# %% Question 6

alpha = 0.001
revenue = 16
cost = 10

cont_disc_saver = []
for thold in range(5, 9):
    cont_disc_it = []
    for servers in range(10, 21):
        gen_matrix_it = gen_matrix(c, l, mu, eta, nu, gamma, k, servers, thold)
        long_run_pbb_vector = compute_long_run_pbb(gen_matrix_it)

        left_side_matrix = alpha * np.ones(shape=(c + 1,)) - gen_matrix_it
        left_side_matrix_inv = np.linalg.inv(left_side_matrix)

        profit_vector = np.array([revenue * min(i, servers) - cost * servers for i in range(0, c+1)])

        cont_disc_profit = np.matmul(left_side_matrix_inv, profit_vector.T)

        cont_disc_it.append(cont_disc_profit)
    cont_disc_saver.append(cont_disc_it)

cont_disc_saver = np.array(cont_disc_saver)
