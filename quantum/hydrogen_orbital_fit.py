from optimization import gaussianObjective
import numpy as np
import matplotlib.pyplot as plt

def gaussian_orbital_3(real_orbital, orbital1, orbital2, orbital3, params, param_grad=False, func_grad=False):
    if param_grad:
        grads = np.array([orbital1.gradient(params[0:2]),
                          orbital2.gradient(params[2:4]),
                          orbital3.gradient(params[4:6])])
        residual = orbital1.function(params[0:2]) + orbital2.function(params[2:4]) + orbital3.function(params[4:6]) - real_orbital
        gradient = np.sum(2 * grads * residual, axis=2).flatten()
        return gradient / np.linalg.norm(gradient)
    elif func_grad:
        return orbital1.gradient_wrt_r(params[0:2]) + orbital2.gradient_wrt_r(params[2:4]) + orbital3.gradient_wrt_r(params[4:6])
    return orbital1.function(params[0:2]) + orbital2.function(params[2:4]) + orbital3.function(params[4:6])

radius = np.linspace(0, 10, 100)

a = gaussianObjective(radius)
b = gaussianObjective(radius)
c = gaussianObjective(radius)

data = a.function([7, 3]) + a.function([2, 1]) + a.function([0, 3])

initial_guess = np.array([2, 0.1, 4, 0.1, 6, 0.1])

tol = 0.1
cost = np.inf
iterlim = 1000
learning_rate = 1
i = 0
while cost > tol and (i := i + 1) < iterlim:
    initial_guess += learning_rate * gaussian_orbital_3(data, a, b, c, initial_guess, param_grad=True)
    cost = np.sum((gaussian_orbital_3(data, a, b, c, initial_guess) - data) ** 2)

print(cost)
print(initial_guess)
plt.plot(radius, gaussian_orbital_3(data, a, b, c, initial_guess))
plt.plot(radius, data)
plt.show()
