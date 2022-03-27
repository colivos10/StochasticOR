import numpy as np
from scipy.stats import poisson

# Create transition matrix

def create_transition_matrix(s_order, S, lambda_rate):

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

    return probability_matrix

# Create the cost vector

def create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b):

    profit_vector = np.zeros(shape=(S+1,), dtype=float)

    for i in range(S+1):
        if i <= s_order:
            profit_vector[i] = -h * i + r * (np.sum([poisson.pmf(k, lambda_rate) * k for k in range(0, S)]) + S * (
                        1 - poisson.cdf(S - 1, lambda_rate))) - b * (1 - poisson.cdf(S, lambda_rate)) - K - c * (S - i)
        else:
            profit_vector[i] = -h * i + r * (np.sum([poisson.pmf(k, lambda_rate) * k for k in range(0, i)]) + i * (
                        1 - poisson.cdf(i - 1, lambda_rate))) - b * (1 - poisson.cdf(i, lambda_rate))

    return profit_vector

def long_run_profit(s_order, S, lambda_rate, c, r, K, h, b):
    probability_matrix = create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)

    identity_minus_probability = np.identity(S+1) - probability_matrix

    identity_minus_probability_trans = np.transpose(identity_minus_probability)

    p_tilde = np.vstack([identity_minus_probability_trans[0:-1], np.ones(shape=(S + 1,))])

    ones_vector = np.zeros(shape=(S+1,))
    ones_vector[-1] = 1

    p_tilde_inverse = np.linalg.inv(p_tilde)
    long_run_probabilities = np.matmul(p_tilde_inverse, ones_vector)
    expected_long_run_profit = np.matmul(profit_vector, long_run_probabilities)

    return expected_long_run_profit

def current_profit(s_order, S, lambda_rate, c, r, K, h, b):

    initial_condition_vector = np.zeros(shape=(S + 1,))
    initial_condition_vector[S] = 1

    probability_matrix = create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)
    one_period_profit = np.matmul(np.matmul(initial_condition_vector, probability_matrix),profit_vector)

    return one_period_profit

def compute_long_run_profit(s_order, S, lambda_rate, c, r, K, h, b):

    probability_matrix = create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)

    identity_minus_probability = np.identity(S+1) - probability_matrix
    identity_minus_probability_trans = np.transpose(identity_minus_probability)

    p_tilde = np.vstack([identity_minus_probability_trans[0:-1], np.ones(shape=(S+1,))])
    ones_vector = np.zeros(shape=(S+1,))
    ones_vector[-1] = 1
    p_tilde_inverse = np.linalg.inv(p_tilde)
    long_run_probabilities = np.matmul(p_tilde_inverse, ones_vector)
    expected_long_run_profit = np.matmul(profit_vector, long_run_probabilities)

    return expected_long_run_profit