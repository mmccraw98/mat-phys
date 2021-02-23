from optimization import harmonic_bond
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(os.path.join('data', 'xyz.txt'), 'r') as f:
    xyz_str = f.read()
print(xyz_str)
p1 = np.array([float(val) for val in xyz_str.split(sep='\n')[0].split(sep='\t')[1:]])
p2 = np.array([float(val) for val in xyz_str.split(sep='\n')[1].split(sep='\t')[1:]])
p3 = np.array([float(val) for val in xyz_str.split(sep='\n')[2].split(sep='\t')[1:]])
p4 = np.array([float(val) for val in xyz_str.split(sep='\n')[3].split(sep='\t')[1:]])

# plot initial geometry

# store three values
stored_states = []

OH = harmonic_bond(1, 3)
HH = harmonic_bond(2, 10)

f_tol = 1
learning_rate = 1e-3
potential = np.inf
i = 0
while potential > f_tol:
    # keeping p1 fixed and adjusting the rest
    p2_grad = OH.gradient_wrt_r2(p1, p2) + HH.gradient_wrt_r1(p2, p3)
    p3_grad = HH.gradient_wrt_r2(p2, p3) + OH.gradient_wrt_r1(p3, p4)
    p4_grad = OH.gradient_wrt_r2(p3, p4)

    p2 -= learning_rate * p2_grad
    p3 -= learning_rate * p3_grad
    p4 -= learning_rate * p4_grad

    i += 1
    if i % int(25000 / 3) == 0:
        stored_states.append([p1, p2, p3, p4])
    potential = OH.potential(p1, p2) + HH.potential(p2, p3) + OH.potential(p3, p4)

print(np.sqrt(np.sum((p1-p2)**2)), np.sqrt(np.sum((p2-p3)**2)), np.sqrt(np.sum((p3-p4)**2)))

# plot final geometry and three intermediate geometries (stored_states)
#molecule = stored_states[-1]

def plot_hooh_xyz(molecule, plot_title):
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

print(create_xyz_str([p1, p2, p3, p4], ['H', 'O', 'O', 'H']))