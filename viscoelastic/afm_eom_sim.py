import numpy as np

t = 0.2
dt = 2e-10

sampling_freq = 1 / 50e4

time_log = np.arange(0, t, sampling_freq)
force_log = np.zeros(time_log.shape)
zb_log = np.zeros(time_log.shape)
zc_log = np.zeros(time_log.shape)

approach_vel = 2000  # nm / s
z_step = approach_vel * dt
F_threshold = 5  # nN

k = 2  # nN / nm
Q = 2
f = 75e3  # Hz
R = 1e3  # nm

m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

zb = [1e3, 0]
zc = zb.copy()
vc = [0, 0]
ac = [0, 0]

surface_height = 0.0  # nm

i = 0
while i < int(0.01*t / dt):
    # move cantilever base
    zb[1] = (zb[0] - z_step) * (ac[0] * m <= F_threshold) + \
            (zb[0] + z_step) * (ac[0] * m > F_threshold)

    # calculate tip-sample force
    force = zc[0] * (zc[0] <= surface_height) * 0

    # integrate eom of cantilever tip according to velocity verlet
    zc[1] = zc[0] + vc[0] * dt + 1 / 2 * ac[0] * dt**2
    ac[1] = (force - b * vc[0] - k * zc[0]) / m
    vc[1] = vc[0] + 1 / 2 * (ac[0] + ac[1]) * dt

    # update values for next iteration
    zb[0] = zb[1]
    zc[0] = zc[1]
    ac[0] = ac[1]
    vc[0] = vc[1]

    # log current data
    if i * dt in time_log:  # i * dt % sampling_freq == 0:
        print('Percent Progress: {:.3f}%'.format(100 * i / (t / dt)))
        index = int(i * sampling_freq)
        force_log[index] = ac[0] * m
        zb_log[index] = zb[0]
        zc_log[index] = zc[0]

    # iterate
    i += 1
# end sim when tip sample force is negative

import matplotlib.pyplot as plt
plt.plot(time_log, force_log, label='probe force')
plt.plot(time_log, zb_log, label='base position')
plt.plot(time_log, zc_log, label='probe position')
plt.legend()
plt.show()
