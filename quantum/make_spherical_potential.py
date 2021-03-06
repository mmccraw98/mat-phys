import auxilary as hf
import numpy as np
from scipy import ndimage


# add sphere potential
def circle_matx(r):
    A = np.arange(-r, r + 1)**2
    dists = np.sqrt(A[:, None] + A)
    return ((dists - r) < 0.5).astype(int)


# xyz in the form of ID, X, Y, Z
xyz = np.array([row.split(sep=' ') for row in hf.load(r'lammps_simulations\dump_final.polymer.xyz', override_extension='txt').split(sep='\n')[9:] if len(row) > 1])
bonds = hf.load(r'lammps_simulations\bonds.xlsx').values.astype(int)
ids = xyz[:, 0].astype(int)  # only IDS
xy = xyz[:, :3].astype(float)  # ID, X, Y

dr = 0.01

r_sph = 0.5  # radius of the potential surrounding each sphere
r_sph_dscr = int(r_sph / dr)  # discretized potential radius
sph_clone = circle_matx(r_sph_dscr)  # 'image' of circular potential surrounding each sphere

pad = r_sph * 3  # padding to prevent clipping
xy[:, 1] -= np.min(xy[:, 1])
xy[:, 2] -= np.min(xy[:, 2])
xy_r = (xy[:, 1:] + pad) / dr

x_max, x_min = np.max(xy[:, 1]) + pad, np.min(xy[:, 1]) - pad
y_max, y_min = np.max(xy[:, 2]) + pad, np.min(xy[:, 2]) - pad

sim_box = np.zeros((int(np.floor((y_max - y_min) / dr)), int(np.floor((x_max - x_min) / dr))))
X, Y = np.linspace(x_min, x_max, sim_box.shape[1]), np.linspace(y_min, y_max, sim_box.shape[0])

# put the 'image' of the circular potential around each of the sphere's centers
for sx, sy in np.floor(xy_r).astype(int):
    sim_box[sy - r_sph_dscr: sy + r_sph_dscr + 1, sx - r_sph_dscr: sx + r_sph_dscr + 1] += sph_clone
sim_box = np.sign(sim_box)  # set all nonzero positive values to 1 to avoid double counting of overlapping potentials

# form electrodes
electrode_separation, sharpness = int(1 / dr), 0.8  # required separation distance from the electrode and nearest molecule
electrode = ndimage.gaussian_filter(np.abs(sim_box - 1), sigma=electrode_separation)  # get 'inverse' image and smooth
electrode[electrode < sharpness] = 0.0
electrode[electrode >= sharpness] = 1.0
electrode[int(0.1 * sim_box.shape[0]): int(0.9 * sim_box.shape[0])] = 0.0  # only consider the lower and upper boundaries for electroding
sim_box += electrode  # put the electrodes into the sim box

# form initial wavefunction (including the electrode)
width = 1
psi_0 = ndimage.gaussian_filter(sim_box, sigma=width)

# flip the values of the potential conducting space -> 0 potential
sim_box = abs(sim_box - 1)

hf.safesave((sim_box, X, Y), r'lammps_simulations\potential_field.pkl', overwrite=True)

import matplotlib.pyplot as plt
plt.imshow(sim_box, aspect='auto', cmap=plt.get_cmap('Greys'))
plt.show()