import helperfunctions as hf
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage

potential = hf.load(r'lammps_simulations\potential_field.pkl')
psi_0 = hf.load(r'lammps_simulations\psi_init.pkl')
X_axis = hf.load(r'lammps_simulations\x_axis.pkl')
Y_axis = hf.load(r'lammps_simulations\y_axis.pkl')


class ADISchrodinger:
    def __init__(self, potential, psi_0, X_axis, Y_axis, dr, dt):  #@TODO introduce physical constants
        self.V = potential * 10000
        self.psi = psi_0
        self.X_axis, self.Y_axis = X_axis, Y_axis
        self.X, self.Y = np.meshgrid(self.X_axis, self.Y_axis)
        self.dr = dr
        self.dt = dt
        self.C = 2 * self.dt / 1j
        self.D = 1j * self.dt / self.dr ** 2
        # calling the class methods to calculate initial values
        self.psi_normalize()
        self.lhsMatrixA(self.Y_axis)
        self.rhsMatrixA(self.X_axis)

    def psi_magnitude(self, psi_i):
        return np.real(np.multiply(np.conjugate(psi_i), psi_i))

    def psi_normalize(self):
        self.psi /= np.sum(self.psi_magnitude(self.psi))

    def momentum(self):
        def first_deriv(f):
            return ndimage.gaussian_filter(f, sigma=5, order=1, mode='wrap')
        return np.multiply(np.conjugate(self.psi), first_deriv(self.psi)) - np.multiply(self.psi, first_deriv(np.conjugate(self.psi)))

    def lhsMatrixA(self, space_axis):
        main_diag = (1 + 2 * self.D) * np.ones((1, space_axis.size ** 2))
        off_diag = - self.D * np.ones((1, space_axis.size ** 2 - 1))
        self.LHS = sparse.diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(space_axis.size ** 2, space_axis.size ** 2)).toarray()

    def rhsMatrixA(self, space_axis):  #@TODO add the potential
        main_diag = (1 - 2 * self.D + self.C * 0) * np.ones((1, space_axis.size ** 2))
        off_diag = - self.D * np.ones((1, space_axis.size ** 2 - 1))
        self.RHS = sparse.diags([main_diag, off_diag, off_diag], [0, -1, 1], shape=(space_axis.size ** 2, space_axis.size ** 2)).toarray()

    def solve(self):
        t = 0
        while (t := t + self.dt) < 10 * self.dt:
            # advance 1/2
            b1 = np.flipud(self.psi).reshape(self.psi.size, 1)
            sol = np.linalg.solve(self.LHS, np.matmul(self.RHS, b1))
            self.psi = np.flipud(sol).reshape(self.psi.shape)

            # advance remaining 1/2
            b2 = np.flipud(self.psi).reshape(self.psi.size, 1)
            sol = np.linalg.solve(self.LHS, np.matmul(self.RHS, b2))
            self.psi = np.flipud(sol).reshape(self.psi.shape)

    def solve_forward_euler(self):
        t = 0
        while (t := t + self.dt) < 10 * self.dt:
            psi_next = self.psi.copy()
            for i in range(1, self.X_axis.size - 1):
                for j in range(1, self.Y_axis.size - 1):
                    psi_next[i, j] = self.psi[i, j] + self.D * (self.psi[i - 1, j] + self.psi[i + 1, j] + self.psi[i, j - 1] + self.psi[i, j + 1] - 4 * self.psi[i, j]) + self.C * self.V[i, j] * self.psi[i, j]
            self.psi = psi_next

    def plot_pdf(self, psi_i):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(self.X, self.Y, self.psi_magnitude(psi_i), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        plt.tight_layout()
        plt.show()


a = ADISchrodinger(potential, psi_0, X_axis, Y_axis, 0.1, 0.1)
a.solve_forward_euler()
b = a.momentum()
a.plot_pdf(a.psi)

