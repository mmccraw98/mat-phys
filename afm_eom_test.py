import numpy as np
import helperfunctions as hf
from viscoelastic_models import maxwell_coeffs, maxwell_tip_sample_force_lee_and_radok
import matplotlib.pyplot as plt

# time settings
dt = 2e-10  # s SETTING berkin suggests mHz
tF = 0.2  # s SETTING
tI = 0  # s

# array settings based on memory
max_array_length = 5000000  # SETTING
length_full = int(tF / dt)
num_chunks = 10 #int(length_full / max_array_length)

# defining chunked arrays
time = np.arange(tI, max_array_length * dt, dt)

zb = np.zeros(time.shape)

zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

ht = np.zeros(time.shape)

# experiment settings - sampling of data
sampling_freq = 50e3  # SETTING
get_every = int(1 / (sampling_freq * dt))

# experimental datastreams
exp_tip_pos = np.zeros(int(tF * sampling_freq))
exp_base_pos = np.zeros(int(tF * sampling_freq))
exp_deflection = np.zeros(int(tF * sampling_freq))
exp_time = np.zeros(int(tF * sampling_freq))

approach_vel = 2000  # nm / s SETTING
z_step = approach_vel * dt

F_threshold = 50  # nN SETTING - 50
triggered = False

# cantilever settings
k = 2  # nN / nm SETTING
Q = 100  # SETTING berkin says 100-300
f = 75e3  # Hz SETTING
R = 500  # nm SETTING for larger R, use a smaller Q (closer to 100) berkin uses tips around 10-100 nm
alpha = 16 * np.sqrt(R) / 3

m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

# defining initial conditions of the cantilever
zt[0] = 100  # nm SETTING
zb[0] = zt[0]

# creating the material matrices and coefficients
surface_height = 0.0
model_params = {'Ee': 1e4*1e-9, 'arms': [{'E': 1e6*1e-9, 'T': 1e-5}]}  # elastic moduli in nn / nm^2 (divide by 1e9)

hf.tic()
for iteration in range(num_chunks):
    print('{:.2f} % Complete'.format(100 * iteration / num_chunks))
    i = -1  # start at -1 since while loop immediately iterates to i += 1 -> i = 0
    while (i := i + 1) < time.size - 1: #@ TODO: make a 'toy' simulation AFM library???? same backend just smaller numbers -> smaller timestep
                        #@ TODO: make a simulation class that has an assumptions method which prints the assumptions of the simulation
        # move the cantilever base
        if f_ts[i] >= F_threshold and not triggered:
            triggered = True
            z_step *= -1
        if zb[i] > zb[0]:
            break
        zb[i] = zb[i - 1] - z_step  # (-1 + 2 * (any(f_ts >= F_threshold)))

        # calculate surface penetration depth
        ht[i] = (surface_height - zt[i]) * (zt[i] <= surface_height)  # ensured to always be positive, no abs necessary

        # calculate tip-sample force and update the material matrices
        #

        # integrate eom of cantilever tip according to velocity verlet
        zt[i + 1] = zt[i] + vt[i] * dt + 1 / 2 * at[i] * dt ** 2
        at[i + 1] = (-k * zt[i] - b * vt[i] + k * zb[i] + f_ts[i]) / m
        vt[i + 1] = vt[i] + 1 / 2 * (at[i] + at[i + 1]) * dt



    # downsample and log experimental datastreams
    bound_lower, bound_upper = round(time[0] * sampling_freq), round(time[-1] * sampling_freq) + 1
    exp_tip_pos[bound_lower: bound_upper] = zt[::get_every]
    exp_base_pos[bound_lower: bound_upper] = zb[::get_every]
    exp_deflection[bound_lower: bound_upper] = (zt - zb)[::get_every]
    exp_time[bound_lower: bound_upper] = time[::get_every]

    # reset values for next bulk iteration, counts as an iteration step
    time = hf.altspace(time[-1], dt, time.size) + dt
    zb[0] = zb[-1] - z_step
    zt[0] = zt[-1] + vt[-1] * dt + 1 / 2 * at[-1] * dt ** 2
    at[0] = (-k * zt[-1] - b * vt[-1] + k * zb[-1] + f_ts[-1]) / m
    vt[0] = vt[-1] + 1 / 2 * (at[-1] + at[0]) * dt
    ht[0] = (surface_height - zt[0]) * (zt[0] <= surface_height)
    f_ts[0], u_matrix, q_matrix = maxwell_tip_sample_force_lee_and_radok(dt, ht[0], un, qn, u_matrix, q_matrix, alpha)

    # log the run-time
    hf.toc()

plt.plot(exp_time, exp_base_pos, label='base pos')
plt.plot(exp_time, exp_tip_pos, label='tip pos')
plt.plot(exp_time, np.ones(exp_time.shape) * surface_height, '--', label='surface')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.show()

hf.getmemuse()
hf.toc()
