import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from optimization import row2mat, SSEScaledGenMaxwell, maxwell_force, harmonic_shear_response


# t = np.arange(0, 2, 0.0001)
# h = 50 * t / max(t / 2) * (t < max(t) / 2) + 50 * 2 * (1 - t / max(t)) * (t >= max(t) / 2)
t = np.arange(0, 1, 0.0001)
h = 50 * t / max(t)
R = 100e-9

Q_real = np.array([1e5, 1e5, 1e-3])

t_matrix = row2mat(t, Q_real[1::2].size)

f_real = maxwell_force(Q_real, t_matrix, t, h, R)

obj = SSEScaledGenMaxwell(f_real, t_matrix, t, h, R)

Q_final = np.array([1e4, 1e4, 1e-2])

G1_mesh, T1_mesh = np.meshgrid(np.linspace(1e2, 1e9, 50), np.logspace(-6, 0, 50))
cost_surface = np.zeros(G1_mesh.size)
for i, (g1, t1) in enumerate(zip(G1_mesh.ravel(), T1_mesh.ravel())):
    cost_surface[i] = obj.function(np.array([Q_real[0], g1, t1]))
cost_surface = cost_surface.reshape(G1_mesh.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(G1_mesh, np.log10(T1_mesh), cost_surface, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('G1')
ax.set_ylabel('T1')
ax.set_zlabel('Cost')
plt.tight_layout()
plt.show()
