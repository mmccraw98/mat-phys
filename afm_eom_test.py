import numpy as np
import helper_functions as hf
from viscoelastic_models import maxwell_coeffs

# time settings
dt = 2e-10  # s SETTING
tF = 0.2  # s SETTING
tI = 0  # s

# array settings based on memory
max_array_length = 100000  # SETTING
length_full = int(tF / dt)
chunk_length = int(length_full / max_array_length)

# experiment settings
sampling_freq = 1 / 50e4  # SETTING

approach_vel = 2000  # nm / s SETTING
z_step = approach_vel * dt

F_threshold = 5  # nN SETTING

# cantilever settings
k = 2  # nN / nm SETTING
Q = 2  # SETTING
f = 75e3  # Hz SETTING
R = 1e3  # nm SETTING
alpha = 16 * np.sqrt(R) / 3

m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

# defining chunked arrays
time = np.arange(tI, chunk_length * dt, dt)

zb = np.zeros(time.shape)

zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

ht = np.zeros(time.shape)

# defining initial conditions of the cantilever
zt[0] = 1e3  # nm SETTING
zb[0] = zt[0]

# creating the material matrices and coefficients
surface_height = 0
model_params = {'Ee': 1, 'arms': [{'E': 1, 'T': 1e-5}, {'E': 1, 'T': 1e-4}, {'E': 1, 'T': 1e-3}]}

print(maxwell_coeffs(model_params))

quit()
u_arr, q_arr = maxwell_coeffs(model_params)

q_matrix = np.zeros((q_arr.size, time.size))
u_matrix = np.zeros((u_arr.size, time.size))

i = 0
while i < time.size - 1:
    # calculate surface penetration depth
    ht[i] = (surface_height - zt[i]) * (zt[i] <= surface_height)

    # calculate the material matrices
    u_matrix[:, -1]
    f_ts[i] = 0

    # move the cantilever base
    zb[i + 1] = zb[i] - (1 + 2 * (at[0] * m <= F_threshold)) * z_step

    # integrate eom of cantilever tip according to velocity verlet
    zt[i + 1] = zt[i] + vt[i] * dt + 1 / 2 * at[i] * dt ** 2
    at[i + 1] = (-k * zt[i] - b * vt[i] + k * zb[i + 1] + f_ts[i]) / m
    vt[i + 1] = vt[i] + 1 / 2 * (at[i] + at[i + 1]) * dt

    # iterate
    i += 1

hf.get_mem_use()
import matplotlib.pyplot as plt
plt.plot(time, zb)
plt.plot(time, zt)
plt.show()

# num_terms = (model_params['Ee'] is not None) + len(model_params['arms'])
# stiffnessArmsStrings = ['Ee', [ 'E{}'.format(i) for i in range(len(model_params['arms']))]]
#
# # making the polynomial terms for u(s)
# if num_terms >= 1:
#     # make the s0 term
#     s0_u = 1
#     tempStringu = '('
#     for i in range(num_terms):
#         if i == 1 and model_params['Ee'] is not None:
#             continue
#         if i == num_terms:
#             tempStringu = 0



