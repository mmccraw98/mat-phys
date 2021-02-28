import numpy as np
from numpy import array, sum, sqrt, convolve, exp, ones, cos, dot, pi, arccos, kron
from scipy.optimize import minimize
from scipy.integrate import cumtrapz


class GDALRM:
    def __init__(self, learning_rate, momentum, lr_raise, lr_lower, jump_height, iterlim=100, paramdeltalim=0.1,
                 pct_log=0.1, mute=True):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lr_raise = lr_raise
        self.lr_lower = lr_lower
        self.jump_height = jump_height
        self.iterlim = iterlim
        self.paramdeltalim = paramdeltalim
        self.pct_log = pct_log
        self.x_optimum = None
        self.f_optimum = None
        self.mute = mute

    def descend(self, initial_guess, f, g, target_score):
        log_length = int(1 / self.pct_log)
        momentum = self.momentum
        learning_rate = self.learning_rate
        jump_height = self.jump_height
        lr_raise = self.lr_raise
        lr_lower = self.lr_lower
        xi = initial_guess
        xi_prev = xi * 0
        fi = f(xi)
        stopped = False
        count = 0
        paramdelta = np.inf
        x_hist, f_hist = [], []
        while count < self.iterlim and fi > target_score:  # not paramdelta < self.paramdeltalim
            count += 1
            gi = g(xi)
            dxi = momentum * xi_prev - (1 - momentum) * learning_rate * gi
            xi_tent = xi + dxi
            fi_tent = f(xi_tent)
            if fi_tent < (1 + jump_height) * fi:
                fi = fi_tent
                xi_prev = xi
                xi = xi_tent
                paramdelta = abs(np.mean(dxi / xi))
                if stopped:
                    stopped = False
                    momentum = self.momentum
                    learning_rate *= lr_raise
            else:
                stopped = True
                momentum = 0
                learning_rate *= lr_lower
                fi = f(xi)
            if count % log_length == 0:
                print('Max Progress: {}% | Gradient Magnitudes: {}'.format(100 * count / self.iterlim, gi), end='\r')
                x_hist.append(xi)
                f_hist.append(fi)
        if not self.mute:
            print('convergence criteria reached after {} iterations'.format(count), end='\r')
        self.x_optimum = xi
        self.f_optimum = fi
        return x_hist, f_hist


def newton(grad, hess, x_init, epsilon=1e-10, max_iterations=1000):
    x = x_init
    for i in range(max_iterations):
        x = x - np.linalg.solve(hess(x), grad(x))
        if np.linalg.norm(grad(x)) < epsilon:
            print('Converged!' + 20 * ' ', end='\r')
            return x, i + 1
    return x, max_iterations


def gradient_descent(grad, x_init, alpha, max_iter=10):
    x = x_init
    for i in range(max_iter):
        x = x - alpha * grad(x)
    return x


def gradient_descent_momentum(grad, x_init, alpha, beta=0.9, max_iter=10):
    x = x_init
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)
        vc = v/(1+beta**(i+1))
        x = x - alpha * vc
    return x


class ObjectiveFunction:
    def __init__(self, *args):
        self.params = args

    def function(self, X):
        raise NotImplementedError

    def gradient(self, X):
        raise NotImplementedError

    def hessian(self, X):
        raise NotImplementedError


class harmonic_bond_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        kb, r0 = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        return 1 / 2 * kb * (mag_rij - r0) ** 2

    def gradient_wrt_ri(self, ri, rj):
        kb, r0 = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        return - kb * (mag_rij - r0) * (rj - ri) / mag_rij

    def gradient_wrt_rj(self, ri, rj):
        kb, r0 = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        return kb * (mag_rij - r0) * (rj - ri) / mag_rij

    def force_ri(self, ri, rj):
        return - self.gradient_wrt_ri(ri, rj)

    def force_rj(self, ri, rj):
        return - self.gradient_wrt_rj(ri, rj)


class coulomb_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        e0, qi, qj = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        # 322.0637 converts Eq to kcal / mol corresponding with DREIDING
        return 322.0637 * qi * qj / (e0 * mag_rij)

    def gradient_wrt_ri(self, ri, rj):
        e0, qi, qj = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        # 322.0637 converts Eq to kcal / mol corresponding with DREIDING
        return - 322.0637 * qi * qj / (e0 * mag_rij ** 3) * (ri - rj)

    def gradient_wrt_rj(self, ri, rj):
        e0, qi, qj = self.params
        mag_rij = sqrt(sum((ri - rj) ** 2))
        # 322.0637 converts Eq to kcal / mol corresponding with DREIDING
        return 322.0637 * qi * qj / (e0 * mag_rij ** 3) * (ri - rj)

    def force_ri(self, ri, rj):
        return - self.gradient_wrt_ri(ri, rj)

    def force_rj(self, ri, rj):
        return - self.gradient_wrt_rj(ri, rj)


class lennard_jones_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        e, s = self.params
        mag_rij = sqrt(sum(ri - rj) ** 2)
        return 4 * e * ((s / mag_rij) ** 12 - (s / mag_rij) ** 6)

    def gradient_wrt_ri(self, ri, rj):
        e, s = self.params
        mag_rij = sqrt(sum(ri - rj) ** 2)
        return - 24 * e * (2 * (s / mag_rij) ** 12 - (s / mag_rij) ** 6) * (ri - rj) / mag_rij ** 2

    def gradient_wrt_rj(self, ri, rj):
        e, s = self.params
        mag_rij = sqrt(sum(ri - rj) ** 2)
        return 24 * e * (2 * (s / mag_rij) ** 12 - (s / mag_rij) ** 6) * (ri - rj) / mag_rij ** 2

    def force_ri(self, ri, rj):
        return - self.gradient_wrt_ri(ri, rj)

    def force_rj(self, ri, rj):
        return - self.gradient_wrt_rj(ri, rj)


class cosine_angle_ijk(ObjectiveFunction):
    def angle_ijk(self, ri, rj, rk):
        rij = ri - rj
        rkj = rk - rj
        mag_rij = sqrt(sum(rij**2))
        mag_rkj = sqrt(sum(rkj**2))
        return np.arccos(dot(rij, rkj) / (mag_rij * mag_rkj))

    def potential(self, ri, rj, rk):
        kt, t0 = self.params
        rij = ri - rj
        rkj = rk - rj
        mag_rij = sqrt(sum(rij**2))
        mag_rkj = sqrt(sum(rkj**2))
        return 1 / 2 * kt * (dot(rij, rkj) / (mag_rij * mag_rkj) - cos(t0))**2

    def gradient_wrt_ri(self, ri, rj, rk):
        kt, t0 = self.params
        rij = ri - rj
        rkj = rk - rj
        mag_rij = sqrt(sum(rij**2))
        mag_rkj = sqrt(sum(rkj**2))
        unit_rij = rij / mag_rij
        unit_rkj = rkj / mag_rkj
        angle_component = dot(rij, rkj) / (mag_rij * mag_rkj) - cos(t0)
        return kt * angle_component * (unit_rkj - dot(unit_rij, unit_rkj) * unit_rij) / mag_rij

    def gradient_wrt_rj(self, ri, rj, rk):
        return - self.gradient_wrt_ri(ri, rj, rk) - self.gradient_wrt_rk(ri, rj, rk)

    def gradient_wrt_rk(self, ri, rj, rk):
        kt, t0 = self.params
        rij = ri - rj
        rkj = rk - rj
        mag_rij = sqrt(sum(rij ** 2))
        mag_rkj = sqrt(sum(rkj ** 2))
        unit_rij = rij / mag_rij
        unit_rkj = rkj / mag_rkj
        angle_component = dot(rij, rkj) / (mag_rij * mag_rkj) - cos(t0)
        return kt * angle_component * (unit_rij - dot(unit_rij, unit_rkj) * unit_rkj) / mag_rkj

    def force_ri(self, ri, rj, rk):
        return - self.gradient_wrt_ri(ri, rj, rk)

    def force_rj(self, ri, rj, rk):
        return - self.gradient_wrt_rj(ri, rj, rk)

    def force_rk(self, ri, rj, rk):
        return - self.gradient_wrt_rk(ri, rj, rk)


class non_bonded_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        e, s, e0, qi, qj = self.params
        return lennard_jones_ij(e, s).potential(ri, rj) + coulomb_ij(e0, qi, qj).potential(ri, rj)

    def gradient_wrt_ri(self, ri, rj):
        e, s, e0, qi, qj = self.params
        return lennard_jones_ij(e, s).gradient_wrt_ri(ri, rj) + coulomb_ij(e0, qi, qj).gradient_wrt_ri(ri, rj)

    def gradient_wrt_rj(self, ri, rj):
        e, s, e0, qi, qj = self.params
        return lennard_jones_ij(e, s).gradient_wrt_rj(ri, rj) + coulomb_ij(e0, qi, qj).gradient_wrt_rj(ri, rj)

    def force_ri(self, ri, rj):
        return - self.gradient_wrt_ri(ri, rj)

    def force_rj(self, ri, rj):
        return - self.gradient_wrt_rj(ri, rj)


class Rosenbrock(ObjectiveFunction):
    def function(self, X):
        a, b = self.params
        x, y = X
        return (a - x)**2 + b * (y - x**2)**2

    def gradient(self, X):
        a, b = self.params
        x, y = X
        return array([-2 * (a - x) - 4 * x * b * (y - x**2), 2 * b * (y - x**2)])

    def hessian(self, X):
        a, b = self.params
        x, y = X
        return array([[2 - 4 * b * (y - 3 * x**2), -4 * b * x], [-4 * b * x, 2 * b]])


def get_t_matrix(t, n_terms):
    # do once at the beginning of any calculation to improve performance
    return np.tile(t, (n_terms, 1)).T

def maxwell_force(Q_array, t_matrix, t, h, R):  # this is over 100 times faster than the np.convolve method (0.04s vs 1.34s)
    return sqrt(R) * 16 / 3 * cumtrapz((Q_array[0] - sum(Q_array[1::2] / Q_array[2::2]
                                                               * exp(-t_matrix / Q_array[2::2]), axis=1)) * h**(3 / 2), t, initial=0)


class SSEScaledGenMaxwell(ObjectiveFunction):
    def function(self, Q_array):
        force_data, t_matrix, t, h, R = self.params
        return sum((1 + t)**20 * (maxwell_force(Q_array, t_matrix, t, h, R) - force_data)**2, axis=0)


class SSESingleMaxwell(ObjectiveFunction):
    def function(self, Q_array):
        force_data, t_matrix, t, h, R = self.params
        return sum((maxwell_force(Q_array, t_matrix, t, h, R) - force_data)**2, axis=0)

    def gradient(self, Q_array):
        force_data, t_matrix, t, h, R = self.params
        pred_force = array([maxwell_force(Q_array, t_matrix, t, h, R),
                            maxwell_force(Q_array, t_matrix, t, h, R),
                            maxwell_force(Q_array, t_matrix, t, h, R)])
        grad_Q = 16 * sqrt(R) / 3 * array([cumtrapz(h**(3 / 2), t, initial=0),
                                           cumtrapz(- exp(-t / Q_array[2]) / Q_array[2] * h**(3 / 2), t, initial=0),
                                           cumtrapz(- Q_array[1] / Q_array[2]**2 * exp(-t / Q_array[2]) * (t / Q_array[2] - 1) * h**(3 / 2), t, initial=0)])
        return sum((pred_force - force_data) * 2 * grad_Q, axis=1)

    def hessian(self, X):
        print('wrong')
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1 = X * norm_weights
        Rh = 16 * sqrt(R) / 3
        h32 = h**(3/2)
        A = Rh * convolve(ones(t.shape), h32, 'full')[: t.size] * (t[1] - t[0])
        B = Rh * convolve(exp(-t / T1), h32, 'full')[: t.size] * (t[1] - t[0])
        C = Rh * convolve(t * E1 / T1**2 * exp(-t / T1), h32, 'full')[: t.size] * (t[1] - t[0])
        D = Rh * convolve(t / T1**2 * exp(-t / T1), h32, 'full')[: t.size] * (t[1] - t[0])
        E = Rh * convolve(t * (t * exp(-t / T1) - 2 * T1 * exp(-t / T1)) / T1**4, h32, 'full')[: t.size] * (t[1] - t[0])
        force = Rh * convolve(Ee + E1 * exp(-t / T1), h32, 'full')[: t.size] * (t[1] - t[0])
        dp1p1 = 2 * sum(A**2)
        dp1p2 = 2 * sum(A * B)
        dp1p3 = 2 * sum(A * C)
        dp2p2 = 2 * sum(B**2)
        dp2p3 = 2 * sum(force * D + B * C - force_data * D)
        dp3p3 = 2 * sum(force * E + D**2 - force_data * E)
        return array([[dp1p1, dp1p2, dp1p3],
                      [dp1p2, dp2p2, dp2p3],
                      [dp1p3, dp2p3, dp3p3]])


def fit_maxwell_nelder(objective, initial_guess):
    result = minimize(objective.function, initial_guess, method='nelder-mead', options={'maxiter': 10000,
                                                                                        'fatol': 10e-60,
                                                                                        'xatol': 10e-20})
    fit_params = result.x
    return fit_params, result.fun, result.nit

