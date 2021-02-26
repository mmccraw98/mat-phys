import helperfunctions as hf
import pandas as pd
import numpy as np
import optimization as op
import os
import matplotlib.pyplot as plt
import constrNMPymaster.constrNMPy as cNM

def maxwell_force_action(time, h, params):
    Q = params[0] + np.sum(params[1::2] * np.exp((- np.outer(np.ones(params[2::2].size), time).T / params[2::2])), axis=1)
    force = np.sqrt(R) * 16 / 3 * np.convolve(Q, h ** (3 / 2), 'full')[: time.size] * (time[1] - time[0])
    return force

data = pd.read_csv(os.path.join('data', '2_25_data_2000nms_adh.csv'))
time = data.time[data.force > 0].values
force = data.force[data.force > 0].values
h = - data.tip[data.force > 0].values
R = 50e-9
data_model = np.array([1e5, 1e7, 5e-4, 1e4, 1e-2])

double_maxwell_objective = op.SSEDoubleMaxwell(force, time, h, R, 1)
single_maxwell_objective = op.SSESingleMaxwell(force, time, h, R, 1)

#params, _, _ = op.fit_maxwell(single_maxwell_objective, np.array([1e4, 1e5, 1e-3]))
#print(op.fit_maxwell_nelder_mead(double_maxwell_objective, np.array([1e4, 1e5, 1e-3, 1e5, 1e-3])))
guess = np.array([1e3, 1e8, 5e-2])

res = cNM.constrNM(double_maxwell_objective.function, guess, [1e1, 1e1, 2e-5, 1e1, 2e-5], [1e9, 1e9, 1e-1, 1e9, 1e-1],
                   xtol=1e-40, ftol=1e-40, maxiter=50000, full_output=True)
cNM.printDict(res)
