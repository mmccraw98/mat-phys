import numpy as np
import helperfunctions as hf
import viscoelastic_models as vm
import matplotlib.pyplot as plt

hf.tic()

dt = 2e-11
tF = 0.001
time = np.arange(0, tF, dt)

zb = np.zeros(time.shape)
zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
ht = np.zeros(time.shape)
tip_n = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

approach_vel = 1000e-9
dz = approach_vel * dt
initial_pos = 0.5e-9
zt[0] = initial_pos  # nm SETTING
zb[0] = zt[0]
vt[0] = -approach_vel

G = np.array([1e5, 1e4, 1e7])
eta = G[1:] * np.array([1e-4, 1e-1])
x = np.zeros(time.shape)
xd = np.zeros((eta.size, time.size))
vd = np.zeros((eta.size, time.size))

k = 2
Q = 100
f = 75e3
R = 500e-9
m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

a0 = 0.00002
triggered = False
i = 0
while (i := i + 1) < time.size:
    if time[i] > 0.0006 and not triggered:
        triggered = True
    if not triggered:
        zb[i] = zb[i-1] - dz
    else:
        zb[i] = zb[i-1] + dz

    zt[i] = zt[i-1] + vt[i-1] * dt + 1 / 2 * at[i-1] * dt ** 2
    at[i] = (-k * zt[i-1] - b * vt[i-1] + k * zb[i-1] + f_ts[i-1]) / m
    vt[i] = vt[i-1] + 1 / 2 * (at[i-1] + at[i]) * dt

    # calculate the depth
    ht[i] = (x[0] - zt[i]) * (zt[i] < x[0])

    # calculate the deformation as dependent upon the tip profile
    tip_n[i] = -ht[i]

    # contact
    if tip_n[i] < x[i]:
        x[i] = zt[i]
        f_arms = - G[1:] * (x[i] - xd[:, i-1])
        vd[:, i] = - f_arms / eta
        xd[:, i] = xd[:, i-1] + vd[:, i] * dt
        f_ts[i] = - G[0] * x[i] + sum(f_arms)
    # non-contact
    else:
        f_e = - G[0] * x[i-1]
        vd[:, i] = f_e / eta
        xd[:, i] = xd[:, i-1] + vd[:, i] * dt
        x[i] = 1 / sum(G) * sum(G[1:] * (xd[:, i-1] + vd[:, i] * dt))
        f_ts[i] = 0

print(time.shape)
plt.title('positions')
plt.plot(time, zt, label='tip')
plt.plot(time, zb, label='base')
plt.plot(time, x, label='surface')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('pos (m)')
plt.show()
plt.title('force')
plt.plot(time, f_ts)
plt.show()
plt.plot(time, xd[0], label='d0')
plt.plot(time, xd[1], label='d1')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('pos (m)')
plt.show()

hf.toc()