from optimization import MultiGaussianObjective, row2mat
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad

radius = np.linspace(0, 5, 200)

data = 2 * np.exp(- radius)
obj = MultiGaussianObjective(radius, data)

initial_guess = np.array([3.0, 2.0, 2.0, 1.0, 1.0, 0.1])
initial_normalization = np.sqrt(1 / (np.pi ** (3 / 2) * sum(initial_guess[0::2] / initial_guess[1::2] ** (3 / 2))))

maxiter = 10000
res = minimize(obj.sse, x0=initial_guess, method='Nelder-Mead', options={'maxiter': maxiter,
                                                                         'maxfev': maxiter,
                                                                         'xatol': 1e-60,
                                                                         'fatol': 1e-60})

def gauss_func(rad):
    c, a = res.x[0::2], res.x[1::2]
    return np.sum(c * np.exp(- a * row2mat(rad, c.size) ** 2), axis=1)

norm = 1 / quad(gauss_func, a=0, b=np.inf)[0]

c, a = res.x[0::2], res.x[1::2]
r = row2mat(radius, c.size)
energy = 4 * np.pi * norm ** 2 * np.sum(((c**2*a)*(r**2*np.exp(-2*a*r**2)-2*a*r**4*np.exp(-2*a*r**2))-(c**2*r*np.exp(-2*a*r**2))), axis=1)

print('Normalization Constant: {}'.format(norm))
print('Final Parameters: {}'.format(res.x))
print('Final Cost: {}'.format(res.fun))
print('Energy Minimum: {} Hartree'.format(np.min(energy)))
print('Location of Energy Minimum: {} Bohr Radii'.format(radius[np.argmin(energy)]))

def gauss_func(rad):
    c, a = res.x[0::2], res.x[1::2]
    return norm * np.sum(c * np.exp(- a * row2mat(rad, c.size) ** 2), axis=1)

for r_f in np.linspace(2.5, 5, 1000):
    integral = quad(gauss_func, a=0, b=r_f)[0]
    if integral >= 0.95:
        r_val = integral
        break

print('Shell of 95% Electron Density: {} Bohr Radii'.format(r_f))

plt.title('Gaussian Fit and Real Radial Orbitals')
plt.plot(radius, initial_normalization * obj.function(initial_guess), label='Initial')
plt.plot(radius, norm * obj.function(res.x), label='Final', linestyle='-.')
plt.plot(radius, norm * data, label='Real', alpha=0.5)
plt.xlabel('Radial Distance (Bohr Radii)')
plt.ylabel('Wavefunction Amplitude')
plt.legend()
plt.grid()
plt.show()

plt.title('Radial Energy Function of Gaussian Fit Orbital')
plt.plot(radius, energy)
plt.xlabel('Radial Distance (Bohr Radii)')
plt.ylabel('Wavefunction Energy')
plt.grid()
plt.show()
