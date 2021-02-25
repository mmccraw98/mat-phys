import numpy as np
import helperfunctions as hf
import viscoelastic_models as vm
import matplotlib.pyplot as plt

hf.tic()

dt = 1e-10
tF = 0.00001
time = np.arange(0, tF, dt)

zb = np.zeros(time.shape)
zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
ht = np.zeros(time.shape)
tip_n = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

k = 2
Q = 50
f = 75e3
R = 50e-9
m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

approach_vel = 1000e-9
dz = approach_vel * dt
initial_pos = 1e-9
zt[0] = initial_pos  # nm SETTING
zb[0] = zt[0]
vt[0] = -approach_vel

model = {'Ee': 1e6, 'arms': [{'E': 1e8, 'T': 5e-1}]}

un, qn = vm.maxwell_coeffs(model)

u_matrix = np.zeros((2, un.size))
q_matrix = np.zeros((2, qn.size))

triggered = False
i = 0
while (i := i + 1) < time.size:
    if i % int(0.1 * time.size) == 0:
        print('Percent Complete: {}%'.format(100 * i / time.size))
    if not triggered:
        zb[i] = zb[i-1] - dz
    else:
        zb[i] = zb[i-1] + dz

    zt[i] = zt[i-1] + vt[i-1] * dt + 1 / 2 * at[i-1] * dt ** 2
    at[i] = (-k * zt[i-1] - b * vt[i-1] + k * zb[i-1] + f_ts[i-1]) / m
    vt[i] = vt[i-1] + 1 / 2 * (at[i-1] + at[i]) * dt

    ht[i] = - zt[i] * (zt[i] < 0)

    f_ts[i], u_matrix, q_matrix = vm.maxwell_tip_sample_force_lee_and_radok_differential_formulation(dt, ht[i], un, qn, u_matrix, q_matrix, R)

    if f_ts[i] > 5 and not triggered:
        triggered = True
hf.toc()

plt.plot(time, f_ts)
plt.plot(time, zt)
plt.show()