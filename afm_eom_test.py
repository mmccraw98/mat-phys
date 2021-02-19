import numpy as np
import helperfunctions as hf
from viscoelastic_models import maxwell_coeffs, maxwell_tip_sample_force_lee_and_radok
import matplotlib.pyplot as plt

hf.tic()

# time settings
dt = 2e-10  # s SETTING berkin suggests mHz
tF = 0.2  # s SETTING
tI = 0  # s

# array settings based on memory
max_array_length = 2500000  # SETTING
length_full = int(tF / dt)
num_chunks = int(length_full / max_array_length)

# experiment settings
sampling_freq = 1 / 50e4  # SETTING

approach_vel = 2000  # nm / s SETTING
z_step = approach_vel * dt

F_threshold = 0.1  # nN SETTING - 50

# cantilever settings
k = 2  # nN / nm SETTING
Q = 2  # SETTING berkin says 100-300
f = 75e3  # Hz SETTING
R = 1000  # nm SETTING for larger R, use a smaller Q (closer to 100) berkin uses tips around 10-100 nm
alpha = 16 * np.sqrt(R) / 3

m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

# defining chunked arrays
time = np.arange(tI, max_array_length * dt, dt)

zb = np.zeros(time.shape)

zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

ht = np.zeros(time.shape)

# defining initial conditions of the cantilever
zt[0] = 1  # nm SETTING
zb[0] = zt[0]

# creating the material matrices and coefficients
surface_height = 0.9
model_params = {'Ee': 1e3, 'arms': [{'E': 1e3, 'T': 1e-3}]}

un, qn = maxwell_coeffs(model_params)

q_matrix = np.zeros((2, qn.size))
u_matrix = np.zeros((2, un.size))

i = -1  # start at -1 since while loop immediately iterates to i += 1 -> i = 0
while (i := i + 1) < time.size - 1: #@ TODO: make a 'toy' simulation AFM library???? same backend just smaller numbers -> smaller timestep
                    #@ TODO: make a simulation class that has an assumptions method which prints the assumptions of the simulation
    # move the cantilever base
    zb[i + 1] = zb[i] - (1 + 2 * (f_ts[i] <= F_threshold)) * z_step

    # integrate eom of cantilever tip according to velocity verlet
    zt[i + 1] = zt[i] + vt[i] * dt + 1 / 2 * at[i] * dt ** 2
    at[i + 1] = (-k * zt[i] - b * vt[i] + k * zb[i] + f_ts[i]) / m
    vt[i + 1] = vt[i] + 1 / 2 * (at[i] + at[i + 1]) * dt

    # calculate surface penetration depth
    ht[i] = (surface_height - zt[i]) * (zt[i] <= surface_height)  # ensured to always be positive, no abs necessary

    # calculate tip-sample force and update the material matrices
    f_ts[i + 1], u_matrix, q_matrix = maxwell_tip_sample_force_lee_and_radok(dt, ht[i], un, qn, u_matrix, q_matrix, alpha)

plt.plot(time, zb, label='base pos')
plt.plot(time, zt, label='tip pos')
plt.plot(time, zt - zb, label='force')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.show()

plt.plot(time, ht, label='indentation')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.show()

hf.getmemuse()
hf.toc()

print(np.where(ht > 0.0)[0].shape, ht.shape, 'proportion of indentation locations')
print(np.mean(ht), 'ht')
print(np.mean(zt), 'zt')
print(np.mean(zb), 'zb')
print(np.mean(vt), 'vt')
print(np.mean(at), 'at')
print(np.mean(f_ts), 'f_ts')
print(m, 'm')
print(k, 'k')
print(b, 'b')
