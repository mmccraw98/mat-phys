import numpy as np
import pandas as pd
from optimization import SSESingleMaxwell, SSEDoubleMaxwell, SSETripleMaxwell, SSEQuadMaxwell, SSEQuintMaxwell, fit_maxwell
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helperfunctions import tic, toc
import pandas as pd

data = pd.read_csv('data\\afm-samples\\simulated_afm_visco_data\\TestCondition2-Data.csv')
indentation_point = np.where(np.diff(data['d (m)']) > 0)[0][0]
force_signal, deflection_signal, time = data['F (N)'].values[indentation_point:], data['d (m)'].values[indentation_point:], data['time (s)'].values[indentation_point:]
time = time - time[0]

R = 1e-06
Ee = 1e+04
E1 = 1e+04
T1 = 0.005
E2 = 1e+04
T2 = 0.0005

objective = SSEDoubleMaxwell(force_signal, time, deflection_signal, R, None)

print(fit_maxwell(objective, np.array([Ee, E1, T1, E2, T2])))
