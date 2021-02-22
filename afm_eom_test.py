import numpy as np
import helperfunctions as hf
from viscoelastic_models import f_ts_jeff_williams_gen_maxwell_model
import matplotlib.pyplot as plt

# time settings
dt = 2e-10  # s SETTING berkin suggests mHz
tF = 0.005  # s SETTING
tI = 0  # s

# array settings based on memory
length_full = int(tF / dt)
max_array_length = 50000000  # SETTING, making sure we aren't going too far
max_array_length = max_array_length if max_array_length < length_full else length_full
num_chunks = int(length_full / max_array_length)
print(num_chunks, max_array_length, length_full)

# defining chunked arrays
time = np.arange(tI, max_array_length * dt, dt)

# cantilever info
zb = np.zeros(time.shape)

zt = np.zeros(time.shape)
vt = np.zeros(time.shape)
at = np.zeros(time.shape)
f_ts = np.zeros(time.shape)

ht = np.zeros(time.shape)

# sample info - maxwell model - double arm
# creating the material matrices and coefficients
model_params = {'Ee': 1e4, 'arms': [{'E': 1e6, 'T': 1e-5}]}
G = np.array([1e4, 1e5, 1e7])
eta = G[1:] * np.array([1e-5, 1e-1])  # eta = G * tau

x = np.zeros(time.shape)  # position of the model -> position of the elastic spring and the arm springs (SURFACE HEIGHT)
xd = np.zeros((eta.size, time.size))  # position of the damper in each arm
vd = np.zeros((eta.size, time.size))  # velocity of the damper in each arm

# experiment settings - sampling of data
sampling_freq = 50e3  # SETTING
get_every = int(1 / (sampling_freq * dt))

# experimental datastreams
exp_tip_pos = np.zeros(int(tF * sampling_freq))
exp_base_pos = np.zeros(int(tF * sampling_freq))
exp_deflection = np.zeros(int(tF * sampling_freq))
exp_indentation = np.zeros(int(tF * sampling_freq))
exp_surface_height = np.zeros(int(tF * sampling_freq))
exp_time = np.zeros(int(tF * sampling_freq))

approach_vel = 2000e-9  # nm / s SETTING
z_step = approach_vel * dt

F_threshold = 5e-9  # nN SETTING - 50
triggered = False

# cantilever settings
k = 2  # nN / nm SETTING
Q = 100  # SETTING berkin says 100-300
f = 75e3  # Hz SETTING
R = 500e-9  # nm SETTING for larger R, use a smaller Q (closer to 100) berkin uses tips around 10-100 nm
alpha = 16 * np.sqrt(R) / 3
A = 10e-21  # hamaker constant for AFM tip-sample VdW interaction
a0 = 2e-1  # interatomic distance between AFM tip and sample (~ 2 Angstroms)

m = k / (2 * np.pi * f)**2
b = 2 * np.pi * f * m / Q

# defining initial conditions
initial_pos = 2e-9
zt[0] = initial_pos  # nm SETTING
zb[0] = zt[0]
vt[0] = -approach_vel

base_returned = False

#@TODO change tic toc to logging instead of printing?
#@TODO vectorize the tip-sample force, add adhesion
#@TODO work on berkins rmdr idea
#@TODO still need to transform moduli and time constants to stiffnesses and viscosities, respectively
#@TODO turn the model params variable into a class object which can convert between the modulus parameters and stiffnesses
#@TODO turn simulation into a class and add assumption methods
#@TODO turn turn tip sample force into a class
#@TODO this in julia
#@TODO use scipy sparse matrices or some other fast data type
#@TODO gpu computing, numba (jit), parallel computing

# page 68 and section c.3 of jeff c williams thesis

hf.tic()
for iteration in range(num_chunks):
    if base_returned:  # don't do anymore calculations, the base has returned to its original position
        break
    print('{:.2f} % Complete'.format(100 * iteration / num_chunks))
    i = 0  # start at -1 since while loop immediately iterates to i += 1 -> i = 0
    while (i := i + 1) < time.size - 1: #@ TODO: make a 'toy' simulation AFM library???? same backend just smaller numbers -> smaller timestep
                        #@ TODO: make a simulation class that has an assumptions method which prints the assumptions of the simulation
        # calculate surface penetration depth
        ht[i] = (x[0] - zt[i]) * (zt[i] <= x[0])  # ensured to always be positive, no abs necessary

        # integrate eom of cantilever tip according to velocity verlet
        zt[i + 1] = zt[i] + vt[i] * dt + 1 / 2 * at[i] * dt ** 2
        at[i + 1] = (-k * zt[i] - b * vt[i] + k * zb[i] + f_ts[i]) / m
        vt[i + 1] = vt[i] + 1 / 2 * (at[i] + at[i + 1]) * dt

        # calculate tip-sample force for time i + 1 and update the surface position
        f_ts, x, xd, vd = f_ts_jeff_williams_gen_maxwell_model(i=i, zt=zt, x=x, xd=xd, vd=vd, force=f_ts, G=G, eta=eta,
                                                               R=R, dt=dt)

        if np.isnan(f_ts[i + 1]):
            print('f', f_ts[i-2:i+2])
            print('zt', zt[i-2:i+2])
            print('x', x[i-2:i+2])
            print('xd', xd[i-2:i+2])
            print('vd', vd[i-2:i+2])
            print('dist', zt[i-2:i+2] - x[i-2:i+2])
            break

        # move the cantilever base
        if f_ts[i + 1] >= F_threshold and not triggered:
            triggered = True
            z_step *= -1
        zb[i + 1] = zb[i] - z_step  # (-1 + 2 * (any(f_ts >= F_threshold)))
        if zb[i + 1] > initial_pos:
            base_returned = True

    # downsample and log experimental datastreams
    # bound_lower, bound_upper = round(time[0] * sampling_freq), round(time[-1] * sampling_freq) + 1
    # exp_tip_pos[bound_lower: bound_upper] = zt[::get_every]
    # exp_base_pos[bound_lower: bound_upper] = zb[::get_every]
    # exp_deflection[bound_lower: bound_upper] = (zt - zb)[::get_every]
    # exp_indentation[bound_lower: bound_upper] = ht[::get_every]
    # exp_surface_height[bound_lower: bound_upper] = x[::get_every]
    # exp_time[bound_lower: bound_upper] = time[::get_every]

    plt.title('data over time: {}/{}'.format(iteration, num_chunks))
    plt.plot(time, zt, label='tip')
    plt.plot(time, zb, label='base')
    plt.plot(time, x, label='surface')
    plt.legend()
    plt.xlabel('time (s)')
    plt.grid()
    plt.show()

    # reset values for next bulk iteration
    #@TODO change time definition to arange(time now, max array length * dt + time now, dt)
    #@TODO and check if the t_final of that is > tF
    time = hf.altspace(time[-1], dt, time.size) + dt
    zb[0] = zb[-1]
    zt[0] = zt[-1]
    at[0] = at[-1]
    vt[0] = vt[-1]
    ht[0] = ht[-1]
    f_ts[0] = f_ts[-1]

    x[0] = x[-1]
    xd[:, 0] = xd[:, -1]
    vd[:, 0] = vd[:, -1]

    # log the run-time
    hf.toc()

plt.plot(zt, label='z')
plt.plot(x, label='x')
plt.legend()
plt.show()
# plt.plot(exp_time, exp_base_pos, label='base pos')
# plt.plot(exp_time, exp_tip_pos, label='tip pos')
# plt.plot(exp_time, exp_surface_height, '--', label='surface')
# plt.legend()
# plt.grid()
# plt.xlabel('time (s)')
# plt.show()

hf.getmemuse()
hf.toc()
