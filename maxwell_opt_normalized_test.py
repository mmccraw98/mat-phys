import numpy as np
import pandas as pd
from optimization import SSESingleMaxwell, SSEDoubleMaxwell, SSETripleMaxwell, SSEQuadMaxwell, SSEQuintMaxwell, fit_maxwell
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helperfunctions import tic, toc
import pandas as pd

#data = pd.read_csv('data\\afm-samples\\simulated_afm_visco_data\\TestCondition2-Data.csv')
#indentation_point = np.where(np.diff(data['d (m)']) > 0)[0][0]
#force_signal, deflection_signal, time = data['F (N)'].values[indentation_point:], data['d (m)'].values[indentation_point:], data['time (s)'].values[indentation_point:]
#time = time - time[0]

R = 1e-06
Ee = 1e+04
E1 = 1e+04
T1 = 0.005
E2 = 1e+04
T2 = 0.0005

def maxwell_force(model_dict, t, h, R):
    model_stiffness = model_dict['Ee'] + np.sum([arm['E'] * np.exp(-t / arm['T']) for arm in model_dict['arms']], axis=0)
    return np.sqrt(R) * 16 / 3 * np.convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])

time = np.arange(0, 1, 0.01)
force_signal = maxwell_force({'Ee': Ee, 'arms': [{'E': E1, 'T': T1}]},
                             time, time*10, 1)


objective = SSESingleMaxwell(force_signal, time, time*10, R, None)

#print(fit_maxwell(objective, np.array([Ee, E1, T1, E2, T2])))
print(objective.hessian([Ee,E1,T1]))