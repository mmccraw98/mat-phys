import numpy as np
from optimization import SSESingleMaxwell, SSEDoubleMaxwell, SSETripleMaxwell, SSEQuadMaxwell, SSEQuintMaxwell, fit_maxwell
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from helper_functions import tic, toc
import pandas as pd


def maxwell_force(model_dict, t, h, R):
    model_stiffness = model_dict['Ee'] + np.sum([arm['E'] * np.exp(-t / arm['T']) for arm in model_dict['arms']], axis=0)
    return np.sqrt(R) * 16 / 3 * np.convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])


t = np.arange(0, 1, 1/2000)
h = 10e-9 * t / t[-1]  # normalized to meter scale
R = 10e-9 * 1  # normalized to meter scale

true_x = []
single_x, single_cost, single_nit = [], [], []
double_x, double_cost, double_nit = [], [], []
triple_x, triple_cost, triple_nit = [], [], []
quad_x, quad_cost, quad_nit = [], [], []
quint_x, quint_cost, quint_nit = [], [], []

for iterations in range(30):  # test
    print('Percent Complete: {:.1f}%'.format(100*iterations/30))
    sim_params = [np.random.uniform(1e3, 1e9),
                  np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                  np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                  np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                  np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                  np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000)]

    for arms in range(1, 6):  # loop over 1 to 5 arms
        sim_x = sim_params[:1 + 2 * arms]
        sim_x_dict = {'Ee': sim_x[0], 'arms': [{'E': E, 'T': T} for E, T in zip(sim_x[1:][0::2], sim_x[1:][1::2])]}
        sim_force = maxwell_force(sim_x_dict, t, h, R)

        single = SSESingleMaxwell(sim_force, t, h, R)
        double = SSEDoubleMaxwell(sim_force, t, h, R)
        triple = SSETripleMaxwell(sim_force, t, h, R)
        quad = SSEQuadMaxwell(sim_force, t, h, R)
        quint = SSEQuintMaxwell(sim_force, t, h, R)

        for fit_attempt in range(4):  # attempt 3 fits per arm scheme
            guess_params = [np.random.uniform(1e3, 1e9),
                            np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                            np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                            np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                            np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000),
                            np.random.uniform(1e3, 1e9), np.random.randint(1, 20000) / np.random.randint(20000, 1000000)]

            single_xi, single_scorei, single_niti = fit_maxwell(single, guess_params[:1 + 2 * 1])
            double_xi, double_scorei, double_niti = fit_maxwell(double, guess_params[:1 + 2 * 2])
            triple_xi, triple_scorei, triple_niti = fit_maxwell(triple, guess_params[:1 + 2 * 3])
            quad_xi, quad_scorei, quad_niti = fit_maxwell(quad, guess_params[:1 + 2 * 4])
            quint_xi, quint_scorei, quint_niti = fit_maxwell(quint, guess_params[:1 + 2 * 5])

            true_x.append(sim_x)
            single_x.append(single_xi), single_cost.append(single_scorei), single_nit.append(single_niti)
            double_x.append(double_xi), double_cost.append(double_scorei), double_nit.append(double_niti)
            triple_x.append(triple_xi), triple_cost.append(triple_scorei), triple_nit.append(triple_niti)
            quad_x.append(quad_xi), quad_cost.append(quad_scorei), quad_nit.append(quad_niti)
            quint_x.append(quint_xi), quint_cost.append(quint_scorei), quint_nit.append(quint_niti)

data = pd.DataFrame()

data['true_x'] = true_x

data['single_x'] = single_x
data['single_cost'] = single_cost
data['single_nit'] = single_nit

data['double_x'] = double_x
data['double_cost'] = double_cost
data['double_nit'] = double_nit

data['triple_x'] = triple_x
data['triple_cost'] = triple_cost
data['triple_nit'] = triple_nit

data['quad_x'] = quad_x
data['quad_cost'] = quad_cost
data['quad_nit'] = quad_nit

data['quint_x'] = quint_x
data['quint_cost'] = quint_cost
data['quint_nit'] = quint_nit

data.to_csv('nelder_mead_full_maxwell_test_results.csv')





quit()
true_x, final_x, initial_cost, final_cost, iterations, fun = [], [], [], [], [], []
for param_set in range(500):
    print('Percent Complete: {:.1f}%'.format(100 * param_set / 500))
    sim_x = np.array([np.random.uniform(1e3, 1e9), np.random.uniform(1e3, 1e9), np.random.uniform(1/20000, 1)])
    sim_params = {'Ee': sim_x[0], 'arms': [{'E': sim_x[1], 'T': sim_x[2]}]}
    sim_model = maxwell_force(sim_params, t, h, R)
    objective = SSESingleMaxwell(sim_model, t, h, R)
    for guess in range(4):
        x_init = [np.random.uniform(1e3, 1e9), np.random.uniform(1e3, 1e9), np.random.uniform(1/20000, 1)]
        res = minimize(objective.function, x_init, method='nelder-mead', options={'maxiter': 10000, 'fatol': 10e-60})
        true_x.append(sim_x), final_x.append(res.x), initial_cost.append(objective.function(x_init))
        final_cost.append(res.fun), iterations.append(res.nit)

data = pd.DataFrame()
data['true_x'] = true_x
data['final_x'] = final_x
data['initial_cost'] = initial_cost
data['final_cost'] = final_cost
data['iterations'] = iterations
data.to_csv('nelder_mead_full_maxwell_test_results.csv')

quit()

# length = 50
# p1_mesh, p2_mesh = np.meshgrid(np.linspace(1e3, 1e5, length), np.linspace(1/20000, 1, length))
# objective_surface = np.array([objective.function([1e4, p1, p2]) for p1, p2 in zip(p1_mesh.ravel(), p2_mesh.ravel())])
# objective_surface.resize(p1_mesh.shape)
#
# hf = plt.figure()
# ha = hf.add_subplot(111, projection='3d')
# ha.plot_surface(p1_mesh, p2_mesh, objective_surface, color='c', alpha=0.6)
# #ha.scatter(2, 1.5, objective.function([1, 2, 1.5]), c='g', marker='x', s=10e2)
# ha.scatter(1e4, 5e-3, objective.function([1e4, 1e4, 5e-3]), c='g', marker='x', s=10e2)
# ha.set_xlabel('E1')
# ha.set_ylabel('T1')
# ha.set_zlabel('Objective Cost')
# plt.show()