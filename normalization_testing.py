import numpy as np
from optimization import get_t_matrix, SSEScaledGenMaxwell, maxwell_force, harmonic_shear_response
from scipy import optimize
import helperfunctions as hf
import os

Q_vals = np.array([ np.array([1e4, 1e6, 1e-2]),  # 1 arm
                   np.array([3e4, 5e5, 5e-3]),
                   np.array([5e4, 5e4, 5e-4]),

                   np.array([1e4, 1e6, 1e-2, 1e3, 8e-3]),  # 2 arm
                   np.array([3e4, 5e5, 5e-3, 1e7, 3e-2]),
                   np.array([5e4, 5e4, 5e-4, 5e4, 1e-4]),

                   np.array([1e4, 1e6, 1e-2, 1e3, 8e-3, 7e4, 3e-2]),  # 3 arm
                   np.array([3e4, 5e5, 5e-3, 1e7, 3e-2, 3e7, 1e-4]),
                   np.array([5e4, 5e4, 5e-4, 5e4, 1e-4, 2e5, 3e-3]),

                   np.array([1e4, 1e6, 1e-2, 1e3, 8e-3, 7e4, 3e-2, 1e3, 4e-3]),  # 4 arm
                   np.array([3e4, 5e5, 5e-3, 1e7, 3e-2, 3e7, 1e-4, 1e4, 4e-2]),
                   np.array([5e4, 5e4, 5e-4, 5e4, 1e-4, 2e5, 3e-3, 1e6, 4e-4])])  # model relaxance

bounds = ((1e1, 1e9), (1e1, 1e9), (1e-5, 1e-1), (1e1, 1e9), (1e-5, 1e-1), (1e1, 1e9), (1e-5, 1e-1), (1e1, 1e9), (1e-5, 1e-1))

for i, Q_real in enumerate(Q_vals):
    print('Trial {} of {}'.format(i + 1, len(Q_vals)))
    t = np.arange(0, 1, 0.0001)  # second time signal
    t_matrix_sim = get_t_matrix(t, Q_real[2::2].size)  # for speedy calculations (this one is for the sim data NOT THE GUESS)
    h = t / max(t) * 50e-9  # ramp input
    R = 100e-9  # 100 nm tip radius
    force_real = maxwell_force(Q_real, t_matrix_sim, t, h, R)
    ## adds noise -> force_real += hf.gaussian_white_noise(0.1 * np.max(force_real), force_real.shape)
    for num_arms in range(1, 5):
        print('---Fitting Attempt {} of 4'.format(num_arms))
        # collect the right bounds for the current guess scheme
        bound = bounds[: 1 + 2 * num_arms]
        t_matrix = get_t_matrix(t, num_arms)  # for speedy calculations (this one is for the current guess NOT THE SIM)
        obj = SSEScaledGenMaxwell(force_real, t_matrix, t, h, R)  # define the objective function
        hf.tic()  # start the timer
        # do the optimization
        res = optimize.dual_annealing(obj.function, bound, maxiter=10000, local_search_options={'method': 'nelder-mead'})
        # put the results into a format
        results = {'Q_real': Q_real, 'Q_final': res.x, 'cost_final': res.fun, 'run_time': hf.toc(return_numeric=True),
                   'time': t, 'indentation': h, 'tip_radius': R}
        # save the results
        file_name = str(num_arms) + '_arms_guess_' + str(Q_real[2::2].size) + '_arms_real.pkl'
        hf.safesave(results, os.path.join('data', 'fitting_noise_ramp', file_name))
