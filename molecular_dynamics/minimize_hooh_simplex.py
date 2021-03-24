from optimization import harmonic_bond_ij, cosine_angle_ijk, non_bonded_ij, dihedral_ijkl
from auxilary import deg_to_rad, rad_to_deg, safesave
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


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


def plot_hooh_state(state, title=''):
    '''
    plots the geometric state of the hooh molecule
    :param state: numpy array state of the hooh molecule in x, y, z coordinates
    :param title: optional str title of the plot
    :return: plot of the geometric state of the hooh molecule
    '''

    # the below function was taken from stackoverflow to ensure that the 3D plot has even axis scales
    def set_axes_equal(ax):
        '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(state[::3, 0], state[::3, 1], state[::3, 2], c='#636363', s=250)  # hydrogens
    ax.scatter(state[1:-1, 0], state[1:-1, 1], state[1:-1, 2], c='#ff0000', s=500)  # oxygens
    for i in range(state.shape[0] - 1):  # bonds
        ax.plot([state[i, 0], state[i + 1, 0]],
                [state[i, 1], state[i + 1, 1]],
                [state[i, 2], state[i + 1, 2]], c='k')
    plt.title(title)
    ax.set_xlabel('X Distance (Å)')
    ax.set_ylabel('Y Distance (Å)')
    ax.set_zlabel('Z Distance (Å)')
    set_axes_equal(ax)
    plt.show()


# add command line arguments for terminal execution
parser = argparse.ArgumentParser(description='obtain the minimal energy configuration of a hydrogen peroxide (hooh) molecule')
parser.add_argument('--xyz_file', help='path of the xyz input file')
parser.add_argument('--iterlim', help='maximum number of minimization iterations to perform')
parser.add_argument('--tol', help='desired final potential of the molecule')
parser.add_argument('--step_size', help='gradient to parameter step size conversion (larger values will lead to instabilities, smaller values will result in slower execution)')
args = parser.parse_args()

path = args.xyz_file
if path is None:
    exit('no path specified')

if args.iterlim is None:
    iterlim = 100
else:
    iterlim = int(args.iterlim)

if args.tol is None:
    tol = 1e-3
else:
    tol = float(args.tol)

if args.step_size is None:
    step_size = 1e-3
else:
    step_size = float(args.step_size)

# load the file, read it as a string, and close it to save memory
try:
    with open(path, 'r') as f:
        xyz_str = f.read()

except:
    exit('could not find file')

# print initial state
print('Initial Configuration:')
print(xyz_str)

# get the float values of the xyz coordinates from the xyz input string
p1 = np.array([float(val) for val in xyz_str.split(sep='\n')[0].split(sep='\t')[1:]])
p2 = np.array([float(val) for val in xyz_str.split(sep='\n')[1].split(sep='\t')[1:]])
p3 = np.array([float(val) for val in xyz_str.split(sep='\n')[2].split(sep='\t')[1:]])
p4 = np.array([float(val) for val in xyz_str.split(sep='\n')[3].split(sep='\t')[1:]])

initial_state = np.array([p1, p2, p3, p4])

# energy 'stiffnesses' taken in kcal/mol, bond lengths taken in Å
# values taken from the following papers:
# https://pubs.acs.org/doi/pdf/10.1021/jp111284t bond stretch, angle bend, van der waals
# https://arxiv.org/pdf/1907.05796.pdf electrostatic
# forms of the potentials taken from the following paper:
# https://pubs.acs.org/doi/pdf/10.1021/j100389a010 DREIDING
oo = harmonic_bond_ij(290, 1.460)  # oxygen-oxygen harmonic bond (only 1 of these)
ho = harmonic_bond_ij(523, 0.964)  # hydrogen-oxygen harmonic bond (2 of these)
# converting the rotational stiffness for the harmonic angle to the rotational stiffness for the cosine harmonic angle according to DREIDING
cijk = 60
tijk = deg_to_rad(98.5)
hoo = cosine_angle_ijk(cijk / np.sin(tijk) ** 2, tijk)  # hydrogen-oxygen-oxygen angle bending (2 of these)
hh = non_bonded_ij(0.0012, 2.1, 1, 0.41, 0.41)  # lennard-jones and coulombic potential between non-bonded hydrogens (only 1 of these)
hooh = dihedral_ijkl(140, deg_to_rad(111.8))

initial_guess = initial_state.flatten()

def potential(vec_all):
    p1 = vec_all[:3]
    p2 = vec_all[3: 6]
    p3 = vec_all[6: 9]
    p4 = vec_all[9:]
    return ho.potential(p1, p2) + oo.potential(p2, p3) + ho.potential(p3, p4) + hoo.potential(p1, p2, p3) + hoo.potential(p2, p3, p4) + hh.potential(p1, p4) + hooh.potential(p1, p2, p3, p4)

result = minimize(potential, initial_guess, method='nelder-mead', options={'maxiter': 10000,
                                                                                        'fatol': 10e-60,
                                                                                        'xatol': 10e-20})

plot_hooh_state(initial_state, 'Initial State | Potential {:.1f} Kcal/mol'.format(0))
plot_hooh_state(result.x.reshape(initial_state.shape), 'Final State | Potential {:.1f} Kcal/mol'.format(0))
