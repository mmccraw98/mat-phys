import optimization as opt
import numpy as np
from scipy import optimize
from numdifftools import Gradient, Hessian, Jacobian

# keeping all arrays as the same length!
speeds = np.linspace(10, 10000, 10) * 1e-9  # 10nm/s -> 10 um/s approach speed ranges
speeds = np.linspace(1, 100000000, 10) * 1e-10  # lets see if this helps
dt = 1e-3
t = opt.row2mat(np.arange(0, 1, dt), speeds.size)  # matlab could never
h = (t * speeds).T  # ez linalg
t = t.T  # transpose to make it easier
R = np.linspace(10, 1000, speeds.size) * 1e-9  # 10nm - 1 um tip radii
R = np.ones(speeds.shape) * 100e-9  # 100 nm tip radius constant for all
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


def fit_all(obj_func, fit_attempts=10, params_known=False):
    fits = {}
    bounds = ((1e1, 1e9), (1e1, 1e9), (1e-6, 1e0), (1e1, 1e9), (1e-6, 1e0), (1e1, 1e9), (1e-6, 1e0), (1e1, 1e9), (1e-6, 1e0), (1e1, 1e9), (1e-6, 1e0))
    # loop over different numbers of arms
    for n in range(1, 6):
        # loop over multiple fitting attempts
        params, costs = [], []
        for fit_attempt in range(fit_attempts):
            res = optimize.dual_annealing(obj_func, bounds[: 1 + 2 * n], maxiter=1000,
                                          local_search_options={'method': 'nelder-mead'})  #@TODO add sequential fitting
            params.append(res.x), costs.append(res.fun)
        params_final = np.mean(np.array(params)[np.where(costs < 10 ** np.mean(np.log10(costs)))], axis=0)
        fit_report = {'params': params_final, 'cost': obj_func(params_final)}
        if params_known:
            fit_report.update({'harm cost': opt.maxwell_total_modulus_sse(Q_real, params_final, np.logspace(0, 6, 1000))})
        fits.update({'{} arm'.format(n): fit_report})
        return fits
        break

ans = fit_all(obj.test_function, params_known=True)

