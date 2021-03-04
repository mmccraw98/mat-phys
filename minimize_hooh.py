from optimization import harmonic_bond_ij, cosine_angle_ijk, non_bonded_ij
from helperfunctions import deg_to_rad, rad_to_deg, safesave
import numpy as np
import matplotlib.pyplot as plt
import argparse
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

# define minimization terms
i = 0  # iteration step
potential_current = np.inf  # arbitrarily high initial potential value, it can only go downhill from here
potential_historic, states_historic = [], []  # will track the potential and store a few intermediate states
while (i := i + 1) < iterlim and potential_current > tol:
    # calculate the gradients for each atom
    p1_grad = ho.gradient_wrt_ri(p1, p2) + hoo.gradient_wrt_ri(p1, p2, p3) + hh.gradient_wrt_ri(p1, p4)
    p2_grad = ho.gradient_wrt_rj(p1, p2) + oo.gradient_wrt_ri(p2, p3) + hoo.gradient_wrt_rj(p1, p2,p3) + hoo.gradient_wrt_ri(p2, p3, p4)
    p3_grad = oo.gradient_wrt_rj(p2, p3) + ho.gradient_wrt_ri(p3, p4) + hoo.gradient_wrt_rj(p2, p3,p4) + hoo.gradient_wrt_rk(p1, p2, p3)
    p4_grad = ho.gradient_wrt_rj(p3, p4) + hoo.gradient_wrt_rk(p2, p3, p4) + hh.gradient_wrt_rj(p1, p4)

    # update the positions of each atom according to their scaled gradients
    p1 -= step_size * p1_grad
    p2 -= step_size * p2_grad
    p3 -= step_size * p3_grad
    p4 -= step_size * p4_grad

    # update the potential with the new points
    potential_current = ho.potential(p1, p2) + oo.potential(p2, p3) + ho.potential(p3, p4) + hoo.potential(p1, p2, p3) + hoo.potential(p2, p3, p4) + hh.potential(p1, p4)

    # log the potential as the minimization progresses (done in a way to reduce memory cost)
    if len(potential_historic) < 100 or i % 100 == 0:
        potential_historic.append(potential_current)

    # log the state during the period in which it changes the most
    if len(states_historic) < 3 and i != 1:
        states_historic.append({'state': np.array([p1, p2, p3, p4]), 'potential': potential_current})

# report and save the results
print('Minimized to potential {:.1f} Kcal/mol in {} steps\nFinal Configuration:'.format(potential_current, i))
final_xyz_str = create_xyz_str([p1, p2, p3, p4], ['H', 'O', 'O', 'H'])
safesave(final_xyz_str, path.split(sep='.txt')[0] + '_minimized.txt')
print(final_xyz_str)

# plot the potential throughout the minimization process
plt.plot(potential_historic)
plt.grid()
plt.title('Potential Throughout Minimization')
plt.xlabel('Minimization Step')
plt.ylabel('Potential (Kcal/mol)')
plt.show()

# plot the initial, intermediate, and final results
plot_hooh_state(initial_state, 'Initial State | Potential {:.1f} Kcal/mol'.format(potential_historic[0]))
for i, state in enumerate(states_historic):
    plot_hooh_state(state['state'], 'Intermediate State {} / {} | Potential {:.1f} Kcal/mol'.format(i + 1, len(states_historic), state['potential']))
plot_hooh_state(np.array([p1, p2, p3, p4]), 'Final State | Potential {:.1f} Kcal/mol'.format(potential_current))

print('\nCalculated Geometries:')
print('Angle HOO 1: {:.2f}°'.format(rad_to_deg(hoo.angle_ijk(p1, p2, p3))))
print('Angle HOO 2: {:.2f}°'.format(rad_to_deg(hoo.angle_ijk(p2, p3, p4))))
print('Length HO 1: {:.2f} Å'.format(np.sqrt(np.sum((p1-p2)**2))))
print('Length OO:   {:.2f} Å'.format(np.sqrt(np.sum((p2-p3)**2))))
print('Length HO 2: {:.2f} Å'.format(np.sqrt(np.sum((p3-p4)**2))))
print('Length HH: {:.2f} Å'.format(np.sqrt(np.sum((p1-p4)**2))))