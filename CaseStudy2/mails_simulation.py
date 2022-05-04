# %%
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sympy import Piecewise
from sympy.abc import t
import json
import pandas as pd

from CaseStudy2 import cs2_scripts as cs2

# %% Number 2
np.random.seed(100395)
size_emails = Parallel(n_jobs=8)(delayed(cs2.log_normal_variates)(4, 1, i) for i in range(10000))
print(f'The mean of the size of the emails is: {np.mean(size_emails)}')
print(f'The variance of the size of the emails is: {np.var(size_emails)}')

# %%  Number 3

lambda_t = np.array([1, 1, 1, 1, 1, 1, 2, 5, 5, 5, 3, 8, 10, 10, 10, 8, 6, 4, 3, 2, 1, 1, 1, 1, 1])
l = np.linspace(0, 24, 100)
steps_piece_wise_function = [((l >= 0) & (l <= 5)), ((l > 5) & (l <= 6)), ((l > 6) & (l <= 7)), ((l > 7) & (l <= 9)),
                             ((l > 9) & (l <= 10)), ((l > 10) & (l <= 11)), ((l > 11) & (l <= 12)),
                             ((l > 12) & (l <= 14)), ((l > 14) & (l <= 17)), ((l > 17) & (l <= 20)),
                             ((l > 20) & (l <= 24))]

return_lambda_function = [lambda l: 1, lambda l: l - 4, lambda l: 3 * l - 16, lambda l: 5,
                          lambda l: -2 * l + 23, lambda l: 5 * l - 47, lambda l: 2 * l - 14, lambda l: 10,
                          lambda l: -2 * l + 38, lambda l: -l + 21, lambda l: 1]

plt.scatter([i for i in range(0, 25)], lambda_t)
plt.plot(np.linspace(0, 24, 100), np.piecewise(l, steps_piece_wise_function, return_lambda_function))
plt.ylim(ymin=0)
plt.xlim(xmin=0, xmax=24)
plt.grid(visible=True, alpha=0.5, zorder=0)
plt.xticks([i for i in range(0, 25)])
plt.yticks([i for i in range(0, 11)])
plt.xlabel('t')
plt.ylabel(r'$\lambda (t)$')
plt.savefig('lambda_t.pdf')
plt.show()

# %% Number 4

formula = Piecewise((1, ((t >= 0) & (t < 5))), (t - 4, ((t >= 5) & (t < 6))), (3 * t - 16, ((t >= 6) & (t < 7))),
                    (5, ((t >= 7) & (t < 9))), (-2 * t + 23, ((t >= 9) & (t < 10))),
                    (5 * t - 47, ((t >= 10) & (t < 11))),
                    (2 * t - 14, ((t >= 11) & (t < 12))), (10, ((t >= 12) & (t < 14))),
                    (-2 * t + 38, ((t >= 14) & (t < 17))),
                    (-t + 21, ((t >= 17) & (t < 20))), (1, ((t >= 20) & (t <= 24))))
lambda_star = np.max(lambda_t)
np.random.seed(32822)

# %% Execute simulation
number_arrivals_sim = Parallel(n_jobs=8)(delayed(cs2.non_homogenous_pp)(i, formula, lambda_star) for i in range(1000))

print(f'The mean of the number of arrivals for one day is: {np.mean(number_arrivals_sim)}')
print(f'The variance of the number of arrivals for one day is: {np.var(number_arrivals_sim)}')

# %% Number 5
np.random.seed(32922)
mu = 4
sigma = 1

arrivals_and_emails = Parallel(n_jobs=7)(
    delayed(cs2.non_homogenous_compound_pp)(i, formula, lambda_star) for i in range(100))

# %% Saving file

# save as json
with open('data.json', 'w') as f:
    json.dump(arrivals_and_emails,f)

# %%

# read the file
with open('data.json') as f:
   lst1 = [tuple(x) for x in json.load(f)]

# %% 6.a and 6.b

email_size_total_per_day = [sum(lst1[i][1]) for i in range(len(lst1))]
email_size_sample_mean = np.mean(email_size_total_per_day)
email_size_sample_variance = np.var(email_size_total_per_day)

# %% data frame of total size
df = pd.DataFrame(email_size_total_per_day)

# %%

pbb_greater_10mb = len(df[df[0] > 10000]) / len(df)
pbb_less_6mb = len(df[df[0] < 6000]) / len(df)

# %% percentiles

percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
values_percentiles = [df.quantile(i)[0] for i in percentiles]

# %% Pbb before noon

pbb_before_noon = sum([sum(np.array(lst1[i][0]) >= 12) < sum(np.array(lst1[i][0]) < 12) for i in range(len(lst1))]) / 10000

# %% Pbb

intervals_simulation = [[sum((np.array(lst1[j][0]) < i+1) & (np.array(lst1[j][0]) >= i)) for i in range(0, 24)]
             for j in range(len(lst1))]

# %%
pbb_busy_12_1 = sum(intervals_simulation[i][12] == np.max(intervals_simulation[i]) for i in range(len(lst1))) / len(lst1)