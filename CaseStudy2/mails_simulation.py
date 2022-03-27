import numpy as np
import matplotlib.pyplot as plt

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

# non homogenous poisson process
lambda_star = np.max(lambda_t)

N = 0
t = 0

uniform_variate = np.random.uniform() # Uniform(0,1)
time_arrival_variate = - np.log(1 - uniform_variate) / lambda_star # x_n
t = t + time_arrival_variate # t = t + x_n

uniform_variate_hat = np.random.uniform() # Uniform(0,1)
