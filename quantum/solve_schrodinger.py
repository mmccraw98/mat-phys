import auxilary as hf
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage
from scipy.sparse.linalg import eigs

potential, X_axis, Y_axis = hf.load(r'lammps_simulations\potential_field.pkl')


class schrodinger_2D:
    def __init__(self, potential, X_axis, Y_axis):  #@TODO introduce physical constants
        self.V = potential
        self.X_axis, self.Y_axis = X_axis, Y_axis
        self.X, self.Y = np.meshgrid(self.X_axis, self.Y_axis)
        self.dr = self.X_axis[1] - self.X_axis[0]
        self.ei_states = None
        self.energies = None

    def norm_states(self):
        if self.ei_states is not None:
            self.ei_states /= (np.sum(np.sqrt(np.multiply(np.conjugate(a.ei_states), a.ei_states)), axis=0) * self.dr ** 2)

    def solve(self, num_ei_states=3):
        hbar = 1
        q = 1
        m = 1

        # forming the x derivative matrix - it is a diagonal operator with some offset diagonals
        tmp_diag = np.ones((self.Y_axis.size - 1) * self.X_axis.size)
        # diag either gets the diagonals of a matrix or makes a matrix with a given set of diagonals and an offset (typically 0)
        dx_2 = (- 2) * sparse.dia_matrix(np.diag(np.ones(self.X_axis.size * self.Y_axis.size))) + (1) * sparse.dia_matrix(np.diag(tmp_diag, - self.X_axis.size)) + (1) * sparse.dia_matrix(np.diag(tmp_diag, self.X_axis.size))
        del tmp_diag

        # forming the y derivative matrix - it is a diagonal operator with some offset diagonals
        tmp_diag = np.ones(self.Y_axis.size * self.X_axis.size)
        tmp_offs = np.ones(self.Y_axis.size * self.X_axis.size - 1)
        tmp_offs[self.X_axis.size - 1::self.X_axis.size] = 0
        dy_2 = (- 2) * sparse.diags(tmp_diag, 0) + (1) * sparse.diags(tmp_offs, -1) + (1) * sparse.diags(tmp_offs, 1)
        del tmp_diag, tmp_offs

        # forming the hamiltonian matrix
        H = - hbar ** 2 / (2 * m) * (dx_2 / self.dr ** 2 + dy_2 / self.dr ** 2) + sparse.diags(self.V.ravel() * q, 0)
        del dx_2, dy_2

        # solve the eigenvalue - eigenvector problem
        ei_vals, ei_states = eigs(H, num_ei_states)
        E = ei_vals / q
        del H

        self.ei_states = ei_states
        self.energies = np.real(E)

        # normalize the eigenstates
        self.norm_states()
        print('done solving')


    def plot_pdf(self, ei_state):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, np.real(ei_state.reshape(self.V.shape) ** 2), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('X Distance')
        ax.set_ylabel('Y Distance')
        ax.set_zlabel('Probability')
        plt.tight_layout()
        plt.show()

    def plot_energies(self):
        plt.plot([i for i in range(1, 1 + self.energies.shape[0])], self.energies)
        plt.xlabel('Energy Eigenvalue')
        plt.ylabel('Energy')
        plt.grid()
        plt.show()


a = schrodinger_2D(potential, X_axis, Y_axis)
a.solve()
a.plot_pdf(a.ei_states[:, -2])
a.plot_energies()
