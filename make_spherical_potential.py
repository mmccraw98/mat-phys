import helperfunctions as hf
import numpy as np


# add sphere potential
def vect_circle(r):
    A = np.arange(-r, r + 1)**2
    dists = np.sqrt(A[:, None] + A)
    return ((dists - r) < 0.5).astype(int)


# xyz in the form of ID, X, Y, Z
xyz = np.array([row.split(sep=' ') for row in hf.load(r'C:\Users\Windows\PolymerSimulation\lammps_simulations\dump_final.polymer.xyz', override_extension='txt').split(sep='\n')[9:] if len(row) > 1])
bonds = hf.load(r'C:\Users\Windows\PolymerSimulation\lammps_simulations\bonds.xlsx').values.astype(int)
ids = xyz[:, 0].astype(int)  # only IDS
xy = xyz[:, :3].astype(float)  # ID, X, Y

dr = 0.01

r_sph = 0.5  # radius of the potential surrounding each sphere
r_sph_dscr = int(r_sph / dr)  # discretized potential radius
sph_clone = vect_circle(r_sph_dscr)  # 'image' of circular potential surrounding each sphere

pad = r_sph * 3  # padding to prevent clipping
xy[:, 1] -= np.min(xy[:, 1])
xy[:, 2] -= np.min(xy[:, 2])
xy_r = (xy[:, 1:] + pad) / dr

x_max, x_min = np.max(xy[:, 1]) + pad, np.min(xy[:, 1]) - pad
y_max, y_min = np.max(xy[:, 2]) + pad, np.min(xy[:, 2]) - pad
sim_box = np.zeros((int(np.floor((y_max - y_min) / dr)), int(np.floor((x_max - x_min) / dr))))

# put the 'image' of the circular potential around each of the sphere's centers
for sx, sy in np.floor(xy_r).astype(int):
    sim_box[sy - r_sph_dscr: sy + r_sph_dscr + 1, sx - r_sph_dscr: sx + r_sph_dscr + 1] += sph_clone

sim_box = np.sign(sim_box)

electrode_separation = int(1 / dr)
max_high, min_low = int(0.1 * sim_box.shape[0]), int(0.9 * sim_box.shape[0])
highs, lows = [], []
for col in sim_box.T:
    high, low = 0, 0
    if sum(col) == 0:
        high = max_high
        low = min_low
    else:
        for i, val in enumerate(col):
            if val != 0:
                if high == 0:
                    high = i
                if i > low:
                    low = i
        if high > max_high:
            high = max_high
        if low < min_low:
            low = min_low
    col[low + electrode_separation:] = 1
    col[:high - electrode_separation] = 1

import matplotlib.pyplot as plt
cmap = plt.cm.gray
norm = plt.Normalize(sim_box.min(), sim_box.max())
rgba = cmap(norm(sim_box))
rgba[range(10), range(10), :3] = 1, 0, 0
plt.imshow(rgba, interpolation='nearest')
plt.show()
