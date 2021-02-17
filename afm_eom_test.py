import numpy as np
import helperfunctions as hf
from viscoelastic_models import maxwell_coeffs
import matplotlib.pyplot as plt

hf.tic()

# time settings
dt = 2e-11  # s SETTING berkin suggests mHz
tF = 0.2  # s SETTING
tI = 0  # s

# array settings based on memory
max_array_length = 10000000  # SETTING
length_full = int(tF / dt)
num_chunks = int(length_full / max_array_length)

# experiment settings
sampling_freq = 1 / 50e4  # SETTING

approach_vel = 2000  # nm / s SETTING
z_step = approach_vel * dt

F_threshold = 50  # nN SETTING

# cantilever settings
k = 2  # nN / nm SETTING
Q = 2  # SETTING berkin says 100-300
f = 75e3  # Hz SETTING
R = 1e3  # nm SETTING for larger R, use a smaller Q (closer to 100) berkin uses tips around 10-100 nm
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
surface_height = 0
model_params = {'Ee': 1e4, 'arms': [{'E': 1e4, 'T': 1e-5}]}

un, qn = maxwell_coeffs(model_params)

q_matrix = np.zeros((2, qn.size))
u_matrix = np.zeros((2, un.size))

i = 0
while i < time.size - 1: #@ TODO: make a 'toy' simulation AFM library???? same backend just smaller numbers -> smaller timestep
                    #@ TODO: make a simulation class that has an assumptions method which prints the assumptions of the simulation
    # calculate surface penetration depth
    ht[i] = (surface_height - zt[i]) * (zt[i] <= surface_height)  # ensured to always be positive, no abs necessary

    # calculate the material matrices
    q_matrix[1, 0] = ht[i]**(3 / 2)  # lowest order strain derivative
    for j in range(qn.size - 1):  # higher order strain derivatives
        q_matrix[1, j + 1] = (q_matrix[1, j] - q_matrix[0, j]) / dt
    u_matrix[1, -1] = (alpha * sum(qn * q_matrix[1]) - sum(un[: -1] * u_matrix[0, :-1])) / un[-1]  # highest order stress derivative
    for j in range(2, un.size + 1): # lower order stress derivatives
        u_matrix[1, -j] = u_matrix[0, -j] + u_matrix[1, -j + 1] * dt

    f_ts[i] = u_matrix[1, 0]  # update interaction force with viscoelastic stress

    q_matrix[0] = q_matrix[1]  # save the current q state as the previous q state
    u_matrix[0] = u_matrix[1]  # save the current u state as the previous u state

    # move the cantilever base
    zb[i + 1] = zb[i] - (1 + 2 * (at[0] * m <= F_threshold)) * z_step

    # integrate eom of cantilever tip according to velocity verlet
    zt[i + 1] = zt[i] + vt[i] * dt + 1 / 2 * at[i] * dt ** 2
    at[i + 1] = (-k * zt[i] - b * vt[i] + k * zb[i] + f_ts[i] ) / m
    vt[i + 1] = vt[i] + 1 / 2 * (at[i] + at[i + 1]) * dt

    # iterate and check exit conditions
    if zb[i] > zb[0]:
        break
    i += 1

plt.plot(time, zb, label='base pos')
plt.plot(time, zt, label='tip pos')
plt.plot(time, zt - zb, label='force')
plt.legend()
plt.grid()
plt.xlabel('time (s)')
plt.show()

hf.get_mem_use()
hf.toc()

print(zt - zb)