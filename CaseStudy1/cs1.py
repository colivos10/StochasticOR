# Import libraries
import numpy as np
import pandas as pd
from CaseStudy1 import scripts_cs1 as cs1

# Parameters
N = 100 # dealer capacity
lambda_rate = 3 # arrival rate
c = 50000 # cost per unit
r = 90000 # revenue
K = 100000 # fixed cost
h = 5000 # inventory holding cost
b = 300000 # goodwill cost

# %% Number 3 statement

policies = [[100, 60], [100, 40], [80, 50], [80, 30], [50, 50], [40, 32], [39, 19]]
writer = pd.ExcelWriter('probability_matrices.xlsx', engine ='xlsxwriter') # excel writer
sheet_counter = 1
for i, j in policies:

    S = i # order size
    s_order = j # reorder point

    probability_matrix = cs1.create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = cs1.create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)
    df = pd.DataFrame(np.vstack([probability_matrix, profit_vector]))
    df.to_excel(writer, sheet_name=f'Sheet{sheet_counter}')

    sheet_counter += 1

writer.save()

# %% Number 4 statement

#policies = [[100, 60], [100, 40], [80, 50], [80, 30], [50, 50], [40, 32], [39, 19]]
policies = [[100, 60]]
n_weeks = 12

profits_policies = []
for i, j in policies:
    cum_profit = 0
    S = i # order size
    s_order = j # reorder point

    initial_condition_vector = np.zeros(shape=(S + 1,))
    initial_condition_vector[S] = 1

    probability_matrix = cs1.create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = cs1.create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)

    for k in range(n_weeks): # compute the expected cum profit for the first 12 weeks
        cum_profit = cum_profit + np.matmul(initial_condition_vector, profit_vector)
        print(cum_profit)
        initial_condition_vector = np.matmul(initial_condition_vector, probability_matrix)

    #profits_policies.append(cum_profit)

#pd.DataFrame(profits_policies).to_excel('number4.xlsx')

# %% Number 5 statement

policies = [[100, 60], [100, 40], [80, 50], [80, 30], [50, 50], [40, 32], [39, 19]]
expected_profit = []
for i, j in policies:

    S = i # order size
    s_order = j # reorder point

    probability_matrix = cs1.create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = cs1.create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)

    identity_minus_probability = np.identity(S+1) - probability_matrix
    identity_minus_probability_trans = np.transpose(identity_minus_probability)

    p_tilde = np.vstack([identity_minus_probability_trans[0:-1], np.ones(shape=(S+1,))])
    ones_vector = np.zeros(shape=(S+1,))
    ones_vector[-1] = 1
    p_tilde_inverse = np.linalg.inv(p_tilde)
    long_run_probabilities = np.matmul(p_tilde_inverse, ones_vector)
    expected_long_run_profit = np.matmul(profit_vector, long_run_probabilities)

    expected_profit.append(expected_long_run_profit)

pd.DataFrame(expected_profit).to_excel('number5.xlsx')

# %% Number 6 statement

policies = [[100, 60], [100, 40], [80, 50], [80, 30], [50, 50], [40, 32], [39, 19]]
cum_profit = 0
discounted_expected_profit = []
alpha = 0.98
for i, j in policies:

    S = i # order size
    s_order = j # reorder point

    probability_matrix = cs1.create_transition_matrix(s_order, S, lambda_rate)
    profit_vector = cs1.create_profit_vector(s_order, S, lambda_rate, c, r, K, h, b)

    identity_minus_probability = np.identity(S+1) - probability_matrix
    identity_minus_probability_trans = np.transpose(identity_minus_probability)

    p_tilde = np.vstack([identity_minus_probability_trans[0:-1], np.ones(shape=(S + 1,))])

    ones_vector = np.zeros(shape=(S+1,))
    ones_vector[-1] = 1

    p_tilde_inverse = np.linalg.inv(p_tilde)
    long_run_probabilities = np.matmul(p_tilde_inverse, ones_vector)


    identity_minus_probability_discounted = np.identity(S + 1) - alpha * probability_matrix
    identity_minus_probability_discounted_inv = np.linalg.inv(identity_minus_probability_discounted)

    phi = np.matmul(identity_minus_probability_discounted_inv, profit_vector)

    discounted_long_run_profit = np.matmul(phi, long_run_probabilities)

    discounted_expected_profit.append(discounted_long_run_profit)

pd.DataFrame(discounted_expected_profit).to_excel('number6.xlsx')

# %% Force brute
long_run_profit_opt = -1000000
S_opt = 0
s_opt = 0
profits_policies = []
for S in range(0, N+1):
    for s in range(S + 1):
        print(s, S)
        long_run_profit = cs1.long_run_profit(s, S, lambda_rate, c, r, K, h, b)
        profits_policies.append((S, s, long_run_profit))
        if long_run_profit > long_run_profit_opt:
            long_run_profit_opt = long_run_profit
            S_opt = S
            s_opt = s

# %% Simulated annealing to find the optimal policy

def uniform_movement_operator(x_null, lower_bound, upper_bound,):
    number_movement = (np.random.random() - 0.5) * (upper_bound - lower_bound)
    x_next_movement = x_null + np.int_(number_movement)

    return x_next_movement

def normal_movement_operator(x_null, lower_bound, upper_bound):
    number_movement = (upper_bound - lower_bound) / 6 * np.random.normal()
    x_next_movement = x_null + number_movement

    return x_next_movement

# Set variables' lower and upper bound
S_lower_bound = 0
S_upper_bound = N
s_lower_bound = 0
s_upper_bound = S_upper_bound

# Info Experiments
seed = 0
temp = 100000

np.random.seed(seed)

# Generate initial solution randomly from a uniform distribution
S_0 = np.random.random_integers(low=S_lower_bound, high=S_upper_bound, size=None)
s_0 = np.random.random_integers(low=s_lower_bound, high=S_0, size=None)

# Simulated annealing parameters
temperature_0 = temp
temperature_next = temperature_0

number_iterations = 10
number_moves = 3
alpha = 0.9

S_next = S_0
s_next = s_0

S_final = S_0
s_final = s_0

z_next = cs1.long_run_profit(s_0, S_0, lambda_rate, c, r, K, h, b)

S_temp = 0
s_temp = 0
profit_it = []

for i in range(number_iterations):
    for j in range(number_moves):
        print(i, j)
        # Get S next solution
        S_temp = uniform_movement_operator(S_next, S_lower_bound, S_upper_bound)
        while (S_temp > S_upper_bound) or (S_temp < S_lower_bound):
            S_temp = uniform_movement_operator(S_next, S_lower_bound, S_upper_bound)
        # Get s next solution
        s_temp = np.minimum(np.random.randint(s_lower_bound, S_temp), uniform_movement_operator(s_next, s_lower_bound, S_temp))
        while (s_temp > S_temp) or (s_temp < s_lower_bound):
            s_temp = np.minimum(np.random.randint(s_lower_bound, S_temp), uniform_movement_operator(s_next, s_lower_bound, S_temp))
            print(s_temp, S_temp)
        z_temp = cs1.long_run_profit(s_temp, S_temp, lambda_rate, c, r, K, h, b)

        # For the evaluation if the next IF is false
        n_random = np.random.random()
        exp_fun = np.exp(-(z_next - z_temp)/ temperature_next)

        if z_temp >= z_next:
            S_next = S_temp
            s_next = s_temp
        elif n_random >= exp_fun:
            S_next = S_temp
            s_next = s_temp
        else: # Solution remains the same
            S_next = S_next
            s_next = s_next

        z_next = cs1.long_run_profit(s_next, S_next, lambda_rate, c, r, K, h, b)
        z_final = cs1.long_run_profit(s_final, S_final, lambda_rate, c, r, K, h, b)

        if z_next >= z_final:
            S_final = S_next
            s_final = s_next
    profit_it.append((S_final, s_final, z_final))
    temperature_next = alpha * temperature_next
