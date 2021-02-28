from afm_sim_functions import maxwell_tip_sample_force_lee_and_radok_action_integral_formulation
import numpy as np
import matplotlib.pyplot as plt

time = np.arange(0, 0.8, 1e-3)
depth = 10e-9
h = depth * (time / max(time)) * (time < 0.4) + (depth - depth * (time / max(time))) * (time >= 0.4)
R = 10e-9

model = {'Ee': 1e6, 'arms': [{'E': 1e8, 'T': 5e-1}]}
force = maxwell_tip_sample_force_lee_and_radok_action_integral_formulation(model, time, h, R)

