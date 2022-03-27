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

