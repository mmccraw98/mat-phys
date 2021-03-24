from optimization import MultiGaussianObjective
import numpy as np
import matplotlib.pyplot as plt

radius = np.linspace(0, 10, 200)

obj = MultiGaussianObjective(radius)

a0 = 0.5
data = 1 / (np.sqrt(np.pi) * a0 ** (3 / 2)) * (np.exp(- radius / a0) +
                                               1 / np.sqrt(32) * (2 - radius / a0) * np.exp(- radius / (2 * a0)) +
                                               1 / np.sqrt(32) * radius / a0 * np.exp(- radius / (2 * a0)) +
                                               1 / 8 * radius / a0 * np.exp(- radius / (2 * a0)))

initial_guess = np.array([2, 1.5, 4, 1.5, 6.7, 2])

# tol = 0.1
# cost = np.inf
# iterlim = 1000
# learning_rate = 0.1
# i = 0
# params = initial_guess.copy()
# while (i := i + 1) < iterlim and cost > tol:
#     params -= learning_rate * obj.sse_gradient(params, data)
#     cost = obj.sse(params, data)
#
# print(params)
# print(cost)
plt.plot(radius, data, label='real')
#plt.plot(radius, obj.function(initial_guess), label='initial')
#plt.plot(radius, obj.function(params), label='final')
plt.legend()
plt.grid()
plt.xlabel('Radial Distance (Bohr Radii)')
plt.ylabel('Wavefunction Amplitude')
plt.show()
