# %%
import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy.abc import t
from sympy.plotting import plot
# %%
z_normals = []
x_log = []

for i in range(10000):
    u1 = np.random.uniform()
    u2 = np.random.uniform()

    #box miller
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)

    # log normal
    mu = 4
    sigma = 1

    x = np.exp(mu + sigma * np.random.choice([z1, z2]))
    x_log.append(x)

print(np.mean(x_log), np.var(x_log))

# Poisson
lambda_t = np.array([1, 1, 1, 1, 1, 2, 5, 5, 5, 3, 8, 10, 10, 10, 8, 6, 4, 3, 2, 1, 1, 1, 1, 1])
plt.scatter([i for i in range(24)], lambda_t)
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.xlabel('t')
plt.ylabel(r'$\lambda (t)$')
plt.plot([i for i in range(24)], lambda_t)
plt.show()

#%%  Piece wise function
t = np.linspace(0, 24, 100)
steps_piece_wise_function = [((t >= 0) & (t <= 5)), ((t > 5) & (t <= 6)), ((t > 6) & (t <= 7)), ((t > 7) & (t <= 9)),
                             ((t > 9) & (t <= 10)), ((t > 10) & (t <= 11)), ((t > 11) & (t <= 12)),
                             ((t > 12) & (t <= 14)), ((t > 14) & (t <= 17)), ((t > 17) & (t <= 20)),
                             ((t > 20) & (t <= 24))]

return_lambda_function = [lambda t: 1, lambda t: t - 4, lambda t: 3*t - 16, lambda t: 5,
                          lambda t: -2*t + 23, lambda t: 5*t - 47, lambda t: 2*t - 14, lambda t: 10,
                          lambda t: -2*t + 38, lambda t: -t + 21, lambda t: 1]

plt.scatter(np.linspace(0, 24, 100), np.piecewise(t, steps_piece_wise_function, return_lambda_function))
plt.plot(np.linspace(0, 24, 100), np.piecewise(t, steps_piece_wise_function, return_lambda_function))
plt.show()
# %%
formula = Piecewise((1, ((t >= 0) & (t <= 5))), (t - 4, ((t > 5) & (t <= 6))), (3*t - 16, ((t > 6) & (t <= 7))),
                    (5, ((t > 7) & (t <= 9))), (-2*t + 23, ((t > 9) & (t <= 10))), (5*t - 47, ((t > 10) & (t <= 11))),
                    (2*t - 14, ((t > 11) & (t <= 12))), (10, ((t > 12) & (t <= 14))), (-2*t + 38, ((t > 14) & (t <= 17))),
                    (-t + 21, ((t > 17) & (t <= 20))), (1, ((t > 20) & (t <= 24))))


# %%
# non homogenous poisson process
lambda_star = np.max(lambda_t)

N = 0
t = 0

uniform_variate = np.random.uniform() # Uniform(0,1)
time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star # x_n
t = t + time_arrival_variate # t = t + x_n

uniform_variate_hat = np.random.uniform() # Uniform(0,1)

