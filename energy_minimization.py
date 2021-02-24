from optimization import harmonic_bond_ij, cosine_angle_ijk, non_bonded_ij
from helperfunctions import deg_to_rad, rad_to_deg, tic, toc
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_hooh_xyz(molecule, plot_title, show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, atom in enumerate(molecule):
        if i in [0, 3]: # hydrogen atoms
            color = '#636363'
            size = 25
        else: # oxygen atoms
            color = '#ff0000'
            size = 75
        ax.scatter(atom[0], atom[1], atom[2], c=color, s=size)  # plot atoms
        if i < len(molecule) - 1:  # plot bonds
            ax.plot([molecule[i][0], molecule[i + 1][0]],
                    [molecule[i][1], molecule[i + 1][1]],
                    [molecule[i][2], molecule[i + 1][2]], 'r-', alpha=0.7)
    ax.set_xlabel('X Position (A)')
    ax.set_ylabel('Y Position (A)')
    ax.set_zlabel('Z Position (A)')
    plt.title(plot_title)
    if show:
        plt.show()

def create_xyz_str(molecule, atom_types):
    xyz_str = ''
    for i, atom_and_type in enumerate(zip(molecule, atom_types)):
        atom, type = atom_and_type
        atom_str = type
        for r in atom:
            atom_str += '\t{:.3f}'.format(r)
        xyz_str += atom_str + (i < len(molecule) - 1) * '\n'
    return xyz_str

tic()

with open(os.path.join('data', 'zyx.txt'), 'r') as f:
    xyz_str = f.read()

print(xyz_str)
p1 = np.array([float(val) for val in xyz_str.split(sep='\n')[0].split(sep='\t')[1:]])
p2 = np.array([float(val) for val in xyz_str.split(sep='\n')[1].split(sep='\t')[1:]])
p3 = np.array([float(val) for val in xyz_str.split(sep='\n')[2].split(sep='\t')[1:]])
p4 = np.array([float(val) for val in xyz_str.split(sep='\n')[3].split(sep='\t')[1:]])

# plot initial geometry

# store three values
stored_states = []
# values taken from https://www.researchgate.net/publication/327471625_Simulation_of_the_ion-induced_shock_waves_effects_on_the_transport_of_chemically_reactive_species_in_ion_tracks
# page 3 ^ - bond energy in kcal/mol, length in angstroms, angle in degrees
HO = harmonic_bond_ij(450, 0.9572)  # 2 of these
OO = harmonic_bond_ij(540, 1.453)  # 1 of these
HOO = cosine_angle_ijk(140, deg_to_rad(102.7))  # 2 of these
HOO = cosine_angle_ijk(140, deg_to_rad(90))  # 2 of these

HH = non_bonded_ij(1, 50, 50)  # 1 of these

print(rad_to_deg(HOO.angle_ijk(p1, p2, p3)))
print(rad_to_deg(HOO.angle_ijk(p2, p3, p4)))

p1[1] = 1
p2 = np.array([0.0, 0, 0])
p3 = np.array([1.0, 0, 0])
theta_range = np.arange(0, 4 * np.pi + 0.01, 0.01)
p1_x = np.cos(theta_range)
p1_y = np.sin(theta_range)
angle = []
potential = []
deriv1 = []
deriv2 = []
deriv3 = []
for x, y in zip(p1_x, p1_y):
    p1[0] = x
    p1[1] = y
    angle.append(rad_to_deg(HOO.angle_ijk(p1, p2, p3)))
    potential.append(HOO.potential(p1, p2, p3))
    deriv1.append(HOO.gradient_wrt_ri(p1, p2, p3))
    deriv2.append(HOO.gradient_wrt_rj(p1, p2, p3))
    deriv3.append(HOO.gradient_wrt_rk(p1, p2, p3))
dv1 = np.array(deriv1)
dv2 = np.array(deriv2)
dv3 = np.array(deriv3)

#plt.plot(theta_range, potential)
plt.plot(theta_range, dv1[:,0]+dv2[:,0]+dv3[:,0], label='d1')
#plt.plot(angle, deriv2, label='d2')
#plt.plot(theta_range, dv3[:,2], label='d3')
plt.legend()
plt.show()


# f_tol = 1e-3
# g_tol = 1e-10
# iter_limit = 50000
# learning_rate = 1e-3
# potential = np.inf
# grad_prev = np.inf
# i = 0
# pot = []
# while abs(potential) > f_tol and (i := i + 1) < iter_limit:
#     # keeping p1 fixed and adjusting the rest
#     #p1_grad = HO.gradient_wrt_ri(p1, p2) + HH.gradient_wrt_ri(p1, p2)
#     p2_grad = HO.gradient_wrt_rj(p1, p2) + OO.gradient_wrt_ri(p2, p3) + HOO.gradient_wrt_rj(p1, p2, p3) + HOO.gradient_wrt_ri(p2, p3, p4)
#     p3_grad = OO.gradient_wrt_rj(p2, p3) + HO.gradient_wrt_ri(p3, p4) + HOO.gradient_wrt_rj(p2, p3, p4) + HOO.gradient_wrt_rk(p1, p2, p3)
#     p4_grad = HO.gradient_wrt_rj(p3, p4) + HOO.gradient_wrt_rk(p2, p3, p4)# + HH.gradient_wrt_rj(p1, p4)
#
#     #p1 -= learning_rate * p1_grad
#     p2 -= learning_rate * p2_grad
#     p3 -= learning_rate * p3_grad
#     p4 -= learning_rate * p4_grad
#
#     if i % int(25000 / 3) == 0:
#         stored_states.append([p1, p2, p3, p4])
#     potential = HO.potential(p1, p2) + OO.potential(p2, p3) + HO.potential(p3, p4) \
#                 + HOO.potential(p1, p2, p3) + HOO.potential(p2, p3, p4)# + HH.potential(p1, p4)
#     pot.append(potential)
#
# print(p2_grad, p3_grad, p4_grad)
# print('done after {} iterations'.format(i))
# print('bond lengths', np.sqrt(np.sum((p1-p2)**2)), np.sqrt(np.sum((p2-p3)**2)), np.sqrt(np.sum((p3-p4)**2)))
# print('distance between hydrogens', np.sqrt(np.sum((p1-p4)**2)))
# print('final potential', potential)
# print('angle 123', rad_to_deg(HOO.angle_ijk(p1, p2, p3)))
# print('angle 234', rad_to_deg(HOO.angle_ijk(p2, p3, p4)))
#
# # plot final geometry and three intermediate geometries (stored_states)
# #molecule = stored_states[-1]
#
# print(create_xyz_str([p1, p2, p3, p4], ['H', 'O', 'O', 'H']))
#
# # plot_hooh_xyz(initial_state, 'initial')
# # for i, molecule in enumerate(stored_states):
# #     plot_hooh_xyz(molecule, 'intermediate state {}'.format(i))
# plot_hooh_xyz([p1, p2, p3, p4], 'final', True)
# plt.plot(pot[::100])
# plt.show()
# toc()