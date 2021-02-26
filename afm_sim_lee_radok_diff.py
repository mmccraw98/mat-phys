import numpy as np
import helperfunctions as hf
import viscoelastic_models as vm
import matplotlib.pyplot as plt
import pandas as pd
import os

hf.tic()

dt = 1e-8
tF = 1
time = np.arange(0, tF, dt)

zb = np.zeros(time.shape)
zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
ht = np.zeros(time.shape)
f_ts = np.zeros(time.shape)
f_adh = np.zeros(time.shape)

k = 2
Q = 50
f = 75e3
R = 50e-9
m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q
a0 = 2 * 10 ** -10
# file:///C:/Users/Windows/Downloads/ijms-13-12773.pdf sio2 on cellulose in air
A = 5.9 * 10 ** - 21

approach_vel = 2000e-9
dz = approach_vel * dt
initial_pos = 2e-9
zt[0] = initial_pos  # nm SETTING
zb[0] = zt[0]
vt[0] = -approach_vel

model = {'Ee': 1e5, 'arms': [{'E': 1e7, 'T': 5e-5}, {'E': 1e4, 'T': 1e-2}]}

un, qn = vm.maxwell_coeffs(model)

u_matrix = np.zeros((2, un.size))
q_matrix = np.zeros((2, qn.size))

f_trigger = 50e-9
triggered = False
i = 0
while (i := i + 1) < time.size:
    if i % int(0.1 * time.size) == 0:
        print('Percent Complete: {}%'.format(100 * i / time.size))
        hf.toc()
    if not triggered:
        zb[i] = zb[i-1] - dz
    else:
        zb[i] = zb[i-1] + dz

    zt[i] = zt[i-1] + vt[i-1] * dt + 1 / 2 * at[i-1] * dt ** 2
    at[i] = (-k * zt[i-1] - b * vt[i-1] + k * zb[i-1] + f_ts[i-1]) / m
    vt[i] = vt[i-1] + 1 / 2 * (at[i-1] + at[i]) * dt

    f_adh[i] = - A * R / (6 * ((zt[i] ** 2) * (zt[i] > a0) + (a0 ** 2) * (zt[i] <= a0)))

    ht[i] = - zt[i] * (zt[i] < 0)

    f_ts[i], u_matrix, q_matrix = vm.maxwell_tip_sample_force_lee_and_radok(dt, ht[i], un, qn, u_matrix, q_matrix, R)
    f_ts[i] += f_adh[i]

    if f_ts[i] > f_trigger and not triggered:
        triggered = True
    if zb[i] > zb[0]:
        time = time[:i]
        zb = zb[:i]
        zt = zt[:i]
        vt = vt[:i]
        at = at[:i]
        ht = ht[:i]
        f_ts = f_ts[:i]
        break
hf.toc()

# downsample and log the simulation data
sim_data = pd.DataFrame()
get_every = int(50e3 * tF)  # emulates 50kHz sampling frequency
sim_data['time'] = time[::get_every]
sim_data['force'] = f_ts[::get_every]
sim_data['tip'] = zt[::get_every]
sim_data['base'] = zb[::get_every]
sim_data['indentation'] = ht[::get_every]
sim_data['R'] = R
sim_data['A'] = A
sim_data['k'] = k
sim_data['Q'] = Q
sim_data['f'] = f
sim_data['approach_vel'] = approach_vel
sim_data['Ee'] = model['Ee']
sim_data['E_arms'] = [arm['E'] for arm in model['arms']]
sim_data['T_arms'] = [arm['T'] for arm in model['arms']]
hf.safesave(sim_data, os.path.join('data', '2_25_data_2000nms_adh.csv'))  # save as a csv

# plot interesting values
plt.plot(sim_data.time, sim_data.force, label='force')
plt.plot(sim_data.time, sim_data.tip, label='tip pos')
plt.plot(sim_data.time, sim_data.base, label='base pos')
plt.legend()
plt.grid()
plt.title('simulation results')
plt.xlabel('time (s)')
plt.show()
