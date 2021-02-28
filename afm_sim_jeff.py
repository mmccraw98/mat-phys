import numpy as np
import helperfunctions as hf
import afm_sim_functions as vm
import matplotlib.pyplot as plt

hf.tic()

dt = 1e-7
tF = 0.0012
time = np.arange(0, tF, dt)

zb = np.zeros(time.shape)
zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
ht = np.zeros(time.shape)
tip_n = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

k = 2/1000
Q = 50
f = 75e3
R = 50e-9
m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

approach_vel = 1000e-11
dz = approach_vel * dt
initial_pos = 0.1*approach_vel*0.0002
dmax = approach_vel*0.0002 - 0.1
zt[0] = initial_pos  # nm SETTING
zb[0] = zt[0]
vt[0] = -approach_vel

amax = (abs(R * dmax)) ** 0.5
y_n = np.linspace(0.0, amax, 1000)  # 1D viscoelastic foundation with specified number of elements
dx = y_n[1]  # space step
g_y = y_n ** 2 / R

G = np.array([1e5, 1e4, 1e8])
eta = G[1:] * np.array([1e-4, 1e-2])

G *= dx
eta *= dx

x = np.zeros(time.shape)
xd = np.zeros((eta.size, time.size))
vd = np.zeros((eta.size, time.size))

f_push = np.zeros(time.shape)
f_repel = np.zeros(time.shape)
f_chill = np.zeros(time.shape)


triggered = False
i = 0
while (i := i + 1) < time.size:
    if time[i] > 0.0002 and not triggered:
        triggered = True
    if not triggered:
        zb[i] = zb[i-1] - dz
    else:
        zb[i] = zb[i-1] + dz

    zt[i] = zt[i-1] + vt[i-1] * dt + 1 / 2 * at[i-1] * dt ** 2
    at[i] = (-k * zt[i-1] - b * vt[i-1] + k * zb[i-1] + f_ts[i-1]) / m
    vt[i] = vt[i-1] + 1 / 2 * (at[i-1] + at[i]) * dt

    if i % 100000 == 0:
        print(i / time.size * 100)
    # contact
    if zt[i] < x[i-1]:
        x[i] = zt[i]
        f_arms = - G[1:] * (x[i] - xd[:, i-1])
        vd[:, i] = - f_arms / eta
        xd[:, i] = xd[:, i-1] + vd[:, i] * dt
        f_ts[i] = - G[0] * x[i] + sum(f_arms) # + vdw
        f_push[i] = f_ts[i]
    # non-contact
    else:
        f_e = - G[0] * x[i-1]
        vd[:, i] = f_e / eta
        xd[:, i] = xd[:, i-1] + vd[:, i] * dt
        x_tentative = G[0] / sum(G) * sum(G[1:] * (xd[:, i-1] + vd[:, i] * dt))
        if x_tentative > zt[i]:  # sample has hit the tip
            x[i] = zt[i]
            # calculate the force imparted on the tip from the sample (tentative force in model - real force in model)
            f_arms = - G[1:] * (x_tentative - xd[:, i - 1])
            f_tentative = - G[0] * x_tentative + sum(f_arms)  # + vdw
            f_arms = - G[1:] * (x[i] - xd[:, i - 1])
            f_real = - G[0] * x[i] + sum(f_arms)  # + vdw
            f_ts[i] = -f_real#-(f_tentative - f_real)  # + vdw
            if f_ts[i] < 0:
                f_ts[i] = 0
            f_repel[i] = f_ts[i]
        else:  # sample is legally relaxing
            x[i] = x_tentative
            f_ts[i] = 0 # +vdw
            f_chill[i] = f_ts[i]


print(time.shape)
plt.title('positions')
plt.plot(time, zt, label='tip')
plt.plot(time, zb, label='base')
plt.plot(time, x, label='surface')
arm_force = np.array([G[1:][j] * (xd[j, :-1] + vd[j, 1:] * dt) for j in range(G.size-1)])
plt.plot(time[1:], G[0]*np.sum(arm_force, axis=0)/sum(G), label='non-cont pos')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.ylabel('pos (m)')
plt.show()
i = 3600
j = 4000
plt.plot(f_push[i:j], label='push')
plt.plot(f_repel[i:j], label='repel')
plt.plot(x[i:j]*1e-6, label='surface')
plt.plot(zt[i:j]*1e-6, '--',label='tip')
#plt.plot(f_chill, label='chill')
plt.legend()
plt.show()

hf.toc()