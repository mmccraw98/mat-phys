import numpy as np
import helperfunctions as hf
import afm_sim_functions as asf
import matplotlib.pyplot as plt
import pandas as pd
import os

sim_vals = [{'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
            'model': {'Ee': 1e4, 'arms': [{'E': 1e4, 'T': 0.0005}]},
            'name': 'test_cond_1.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}]},
             'name': 'test_cond_2.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}]},
             'name': 'test_cond_3.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}, {'E': 1e4, 'T': 0.5}]},
             'name': 'test_cond_4.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e4, 'arms': [{'E': 1e4, 'T': 0.0005}]},
             'name': 'test_cond_5.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}]},
             'name': 'test_cond_6.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}]},
             'name': 'test_cond_7.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e4, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}, {'E': 1e4, 'T': 0.5}]},
             'name': 'test_cond_8.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 1e4, 'T': 0.0005}]},
             'name': 'test_cond_9.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}]},
             'name': 'test_cond_10.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}]},
             'name': 'test_cond_11.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 1e4, 'T': 0.0005}, {'E': 1e4, 'T': 0.005}, {'E': 1e4, 'T': 0.05}, {'E': 1e4, 'T': 0.5}]},
             'name': 'test_cond_12.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e5, 'T': 0.0005}]},
             'name': 'test_cond_13.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}]},
             'name': 'test_cond_14.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}]},
             'name': 'test_cond_15.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}, {'E': 5e5, 'T': 0.5}]},
             'name': 'test_cond_16.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e5, 'T': 0.0005}]},
             'name': 'test_cond_17.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}]},
             'name': 'test_cond_18.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}]},
             'name': 'test_cond_19.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}, {'E': 5e5, 'T': 0.5}]},
             'name': 'test_cond_20.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e5, 'T': 0.0005}]},
             'name': 'test_cond_21.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}]},
             'name': 'test_cond_22.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}]},
             'name': 'test_cond_23.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e5, 'T': 0.0005}, {'E': 5e5, 'T': 0.005}, {'E': 5e5, 'T': 0.05}, {'E': 5e5, 'T': 0.5}]},
             'name': 'test_cond_24.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e7, 'T': 0.0005}]},
             'name': 'test_cond_25.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}]},
             'name': 'test_cond_26.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}]},
             'name': 'test_cond_27.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 1e4, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}, {'E': 5e7, 'T': 0.5}]},
             'name': 'test_cond_28.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e7, 'T': 0.0005}]},
             'name': 'test_cond_29.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}]},
             'name': 'test_cond_30.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}]},
             'name': 'test_cond_31.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e5, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}, {'E': 5e7, 'T': 0.5}]},
             'name': 'test_cond_32.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e7, 'T': 0.0005}]},
             'name': 'test_cond_33.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}]},
             'name': 'test_cond_34.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}]},
             'name': 'test_cond_35.csv'},
            {'v': 2e-6, 'k': 2, 'Q': 2, 'f': 7.5e4, 'R': 1e-6,
             'model': {'Ee': 5e7, 'arms': [{'E': 5e7, 'T': 0.0005}, {'E': 5e7, 'T': 0.005}, {'E': 5e7, 'T': 0.05}, {'E': 5e7, 'T': 0.5}]},
             'name': 'test_cond_36.csv'}]

for sim_val in sim_vals:
    print(sim_val['name'])
    sim_val.update({'nu': 0.5})
    approach_vel = sim_val['v']
    k = sim_val['k']
    Q = sim_val['Q']
    f = sim_val['f']
    R = sim_val['R']
    model = sim_val['model']
    name = sim_val['name']

    hf.tic()

    dt = 1e-8
    tF = 1.5
    time = np.arange(0, tF, dt)

    zb = np.zeros(time.shape)
    zt = np.zeros(time.shape)
    vt = np.zeros(time.shape)
    at = np.zeros(time.shape)
    ht = np.zeros(time.shape)
    f_ts = np.zeros(time.shape)
    f_adh = np.zeros(time.shape)

    m = k / (2 * np.pi * f)**2
    b = 2 * np.pi * f * m / Q
    a0 = 2 * 10 ** -10
    # file:///C:/Users/Windows/Downloads/ijms-13-12773.pdf sio2 on cellulose in air
    A = 0#5.9 * 10 ** - 21
    sim_val.update({'hamaker_constant': A})

    dz = approach_vel * dt
    initial_pos = 4e-9
    zt[0] = initial_pos  # nm SETTING
    zb[0] = zt[0]
    vt[0] = -approach_vel

    un, qn = asf.maxwell_coeffs(model)

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

        f_ts[i], u_matrix, q_matrix = asf.maxwell_LR_force_SIM(dt, ht[i], un, qn, u_matrix, q_matrix, R)
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
    sim_data['sim_params'] = ''
    sim_data['sim_params'][0] = sim_val
    hf.safesave(sim_data, os.path.join('data', 'viscoelasticity', name))  # save as a csv

    # plot interesting values
    # plt.plot(sim_data.time, sim_data.force, label='force')
    # plt.plot(sim_data.time, sim_data.tip, label='tip pos')
    # plt.plot(sim_data.time, sim_data.base, label='base pos')
    # plt.legend()
    # plt.grid()
    # plt.title('simulation results')
    # plt.xlabel('time (s)')
    # plt.show()
