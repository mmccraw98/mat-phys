from optimization import harmonic_bond_ij, cosine_angle_ijk, non_bonded_ij
from helperfunctions import deg_to_rad, rad_to_deg, tic, toc
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_hooh_xyz(molecule, plot_title=None, show=True):
    '''
    plot the coordinates of a hooh (hydrogen peroxide) molecule in a 3d plot showing the molecule in color
    red atoms = oxygen, grey atoms = hydrogen, red lines = bonds
    :param molecule: listlike of positional coordinates of the atoms in the molecule
    :param plot_title: optional string argument of the title of the plot
    :param show: optional boolean argument to show the plot or wait for other plots to be added
    :return: plots the molecule
    '''
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
    if plot_title is not None:  # if a title is given, add it
        plt.title(plot_title)
    if show:  # if the plot needs to be shown immediately, show it
        plt.show()


def create_xyz_str(molecule, atom_types):
    '''
    give coordinates of atoms in a molecule and their names, will format to a string in the form of an xyz file
    :param molecule: listlike containing positional coordinates of each atom in the molecule
    :param atom_types: listlike containing string atom types (i.e. atom names)
    :return: string xyz formatted atom type atom coordinates (line by line)
    '''
    xyz_str = ''
    for i, atom_and_type in enumerate(zip(molecule, atom_types)):
        atom, type = atom_and_type
        atom_str = type
        for r in atom:
            atom_str += '\t{:.3f}'.format(r)
        xyz_str += atom_str + (i < len(molecule) - 1) * '\n'
    return xyz_str


path = os.path.join('data', 'xyz.txt')
iterlim = 10000
tol = 1e-3
rate = 1e-3

# load the file, read it as a string, and close it to save memory
with open(path, 'r') as f:
    xyz_str = f.read()

# print initial state
print('Initial Configuration:\n', xyz_str)

p1 = np.array([float(val) for val in xyz_str.split(sep='\n')[0].split(sep='\t')[1:]])
p2 = np.array([float(val) for val in xyz_str.split(sep='\n')[1].split(sep='\t')[1:]])
p3 = np.array([float(val) for val in xyz_str.split(sep='\n')[2].split(sep='\t')[1:]])
p4 = np.array([float(val) for val in xyz_str.split(sep='\n')[3].split(sep='\t')[1:]])

# plot initial state
plot_hooh_xyz([p1, p2, p3, p4], 'Initial Configuration', True)

# define potential terms
# values taken from https://www.researchgate.net/publication/327471625_Simulation_of_the_ion-induced_shock_waves_effects_on_the_transport_of_chemically_reactive_species_in_ion_tracks
oo = harmonic_bond_ij(540, 1.453)  # oxygen-oxygen harmonic bond (only 1 of these)
ho = harmonic_bond_ij(450, 0.9572)  # hydrogen-oxygen harmonic bond (2 of these)
hoo = cosine_angle_ijk(140, deg_to_rad(102.7))  # hydrogen-oxygen-oxygen angle bending (2 of these)
hh = non_bonded_ij()  # lennard-jones and coulombic potential between non-bonded hydrogens (only 1 of these)

# define minimization terms
i = 0  # iteration step
potential_current = np.inf  # arbitrarily high initial potential value, it can only go downhill from here
potential_historic, state_historic = [], []  # will track the potential and store a few intermediate states
while (i := i + 1) < iterlim and potential_current > tol:
    # calculate the gradients for each atom
    p1_grad = ho.gradient_wrt_ri(p1, p2) + hoo.gradient_wrt_ri(p1, p2, p3) # + hh.gradient_wrt_ri(p1, p4)
    p2_grad = ho.gradient_wrt_rj(p1, p2) + oo.gradient_wrt_ri(p2, p3) + hoo.gradient_wrt_rj(p1, p2,p3) + hoo.gradient_wrt_ri(p2, p3, p4)
    p3_grad = oo.gradient_wrt_rj(p2, p3) + ho.gradient_wrt_ri(p3, p4) + hoo.gradient_wrt_rj(p2, p3,p4) + hoo.gradient_wrt_rk(p1, p2, p3)
    p4_grad = ho.gradient_wrt_rj(p3, p4) + hoo.gradient_wrt_rk(p2, p3, p4)  # + hh.gradient_wrt_rj(p1, p4)

    # update the positions of each atom according to their scaled gradients
    p1 -= rate * p1_grad
    p2 -= rate * p2_grad
    p3 -= rate * p3_grad
    p4 -= rate * p4_grad

    # update the potential with the new points
    potential_current = ho.potential(p1, p2) + oo.potential(p2, p3) + ho.potential(p3, p4) + hoo.potential(p1, p2, p3) + hoo.potential(p2, p3, p4)# + hh.potential(p1, p4)

    # log the potential as the minimization progresses (done in a way to reduce memory cost)
    if len(potential_historic) < 100 or i % 100 == 0:
        potential_historic.append(potential_current)

    # log the state during the period in which it changes the most
    if potential_current < 0.8 * sum(potential_historic) / len(potential_historic) and len(state_historic) < 5:
        state_historic.append([p1, p2, p3, p4])

# report the results
print('Minimized to potential ({}) in {} steps\nFinal Configuration:\n'.format(potential_current, i))
final_xyz_str = create_xyz_str([p1, p2, p3, p4], ['H', 'O', 'O', 'H'])
print(final_xyz_str)

# plot the potential throughout the minimization process
plt.plot(potential_historic)
plt.grid()
plt.xlabel('Minimization Step')
plt.ylabel('Potential')
plt.show()

# plot the intermediate states
for j, state in enumerate(state_historic):
    plot_hooh_xyz(state, 'Intermediate Configuration {} / {}'.format(j, len(state_historic)), True)

# plot the final state
plot_hooh_xyz([p1, p2, p3, p4], 'Final Configuration', True)