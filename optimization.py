import numpy as np
from numpy import array, sum, sqrt, convolve, exp, ones, cos, dot, pi, arccos
from scipy.optimize import minimize


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
        print('Undefined!')

    def gradient(self, X):
        print('Undefined!')

    def hessian(self, X):
        print('Undefined!')


class harmonic_bond_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        kb, r0 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return 1 / 2 * kb * (r - r0) ** 2

    def gradient_wrt_ri(self, ri, rj):
        kb, r0 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return - kb * (r - r0) * (rj - ri) / r

    def gradient_wrt_rj(self, ri, rj):
        kb, r0 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return kb * (r - r0) * (rj - ri) / r

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
        return kt * (dot(rij, rkj) / (mag_rij * mag_rkj) - cos(t0)) * (unit_rkj - dot(unit_rij, unit_rkj) * unit_rij) / mag_rij

    def gradient_wrt_rj(self, ri, rj, rk):
        return self.gradient_wrt_ri(ri, rj, rk) + self.gradient_wrt_rk(ri, rj, rk)

    def gradient_wrt_rk(self, ri, rj, rk):
        kt, t0 = self.params
        rij = ri - rj
        rkj = rk - rj
        mag_rij = sqrt(sum(rij ** 2))
        mag_rkj = sqrt(sum(rkj ** 2))
        unit_rij = rij / mag_rij
        unit_rkj = rkj / mag_rkj
        return kt * (dot(rij, rkj) / (mag_rij * mag_rkj) - cos(t0)) * (unit_rij - dot(unit_rij, unit_rkj) * unit_rkj) / mag_rkj

class non_bonded_ij(ObjectiveFunction):
    def potential(self, ri, rj):
        e0, q1, q2 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return q1 * q2 / (4 * pi * e0 * r)

    def gradient_wrt_ri(self, ri, rj):
        e0, q1, q2 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return - q1 * q2 / (4 * pi * e0) * (ri - rj) / r ** 3

    def gradient_wrt_rj(self, ri, rj):
        e0, q1, q2 = self.params
        r = sqrt(sum((ri - rj) ** 2))
        return q1 * q2 / (4 * pi * e0) * (ri - rj) / r ** 3

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


class SSESingleMaxwell(ObjectiveFunction):
    def function(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        return sum((force_predicted - force_data)**2)

    def gradient(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        stiffness_derivs = array([convolve(ones(t.shape), h**(3/2), 'full')[: t.size] * (t[1] - t[0]),
                                  convolve(exp(-t / T1), h**(3/2), 'full')[: t.size] * (t[1] - t[0]),
                                  convolve(E1 * t / T1**2 * exp(-t / T1), h**(3/2), 'full')[: t.size] * (t[1] - t[0])])
        return sum(2 * (force_predicted - force_data) * (16 * sqrt(R) / 3 * stiffness_derivs), axis=1)

    def hessian(self, X):
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


class SSEDoubleMaxwell(ObjectiveFunction):
    def function(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1, E2, T2 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1) + E2 * np.exp(-t / T2)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        return sum((force_predicted - force_data)**2)

    def gradient(self, X):
        raise NotImplementedError

    def hessian(self, X):
        raise NotImplementedError


class SSETripleMaxwell(ObjectiveFunction):
    def function(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1, E2, T2, E3, T3 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1) + E2 * np.exp(-t / T2) + E3 * np.exp(-t / T3)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        return sum((force_predicted - force_data)**2)

    def gradient(self, X):
        raise NotImplementedError

    def hessian(self, X):
        raise NotImplementedError


class SSEQuadMaxwell(ObjectiveFunction):
    def function(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1, E2, T2, E3, T3, E4, T4 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1) + E2 * np.exp(-t / T2) + E3 * np.exp(-t / T3) + E4 * np.exp(-t / T4)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        return sum((force_predicted - force_data)**2)

    def gradient(self, X):
        raise NotImplementedError

    def hessian(self, X):
        raise NotImplementedError


class SSEQuintMaxwell(ObjectiveFunction):
    def function(self, X):
        force_data, t, h, R, norm_weights = self.params
        if norm_weights is None:
            norm_weights = 1
        Ee, E1, T1, E2, T2, E3, T3, E4, T4, E5, T5 = X * norm_weights
        model_stiffness = Ee + E1 * np.exp(-t / T1) + E2 * np.exp(-t / T2) + E3 * np.exp(-t / T3)\
                          + E4 * np.exp(-t / T4) + E5 * np.exp(-t / T5)
        force_predicted = sqrt(R) * 16 / 3 * convolve(model_stiffness, h**(3/2), 'full')[: t.size] * (t[1] - t[0])
        return sum((force_predicted - force_data)**2)

    def gradient(self, X):
        raise NotImplementedError

    def hessian(self, X):
        raise NotImplementedError


def fit_maxwell(objective, initial_guess):
    result = minimize(objective.function, initial_guess, method='nelder-mead', options={'maxiter': 10000,
                                                                                        'fatol': 10e-60})
    fit_params = result.x
    return fit_params, result.fun, result.nit
