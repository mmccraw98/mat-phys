import optimization as opt
import numpy as np
from scipy import optimize
from numdifftools import Gradient, Hessian, Jacobian

# keeping all arrays as the same length!
speeds = np.linspace(10, 10000, 50) * 1e-9  # 10nm/s -> 10 um/s approach speed ranges
dt = 1e-3
t = opt.row2mat(np.arange(0, 1, dt), speeds.size)  # matlab could never
h = (t * speeds).T  # ez linalg
t = t.T  # transpose to make it easier
R = np.linspace(10, 1000, speeds.size) * 1e-9  # 10nm - 1 um tip radii
# defining the model
Q_real = np.array([1e5, 1e5, 1e-3, 1e7, 1e-1])
forces = np.array([opt.maxwell_force_dumb_and_slow(Q_real, time, hi, Ra) for time, hi, Ra in zip(t, h, R)])


# create the dict
params = {'f': forces,
          't': t,
          'h': h,
          'R': R}

# define the objective function
obj = opt.SSE_simultaneous_gen_maxwell(params)

Q_guess = np.array([1e5, 1e3, 1e-1])

#bound = ((1e1, 1e9), (1e1, 1e9), (1e-6, 1e0), (1e1, 1e9), (1e-6, 1e0))
#res = optimize.dual_annealing(obj.test_function, bound, maxiter=1000, local_search_options={'method': 'nelder-mead'})
#print(res.x, res.fun)
