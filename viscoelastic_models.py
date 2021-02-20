from sympy import symbols, denom, expand, collect, numer
from numpy import array, zeros, sqrt


def maxwell_coeffs(model_dict):
    Ee = model_dict['Ee']
    E_arms = array([arm['E'] for arm in model_dict['arms']])  # gets all the elastic stiffnesses for the arms
    T_arms = array([arm['T'] for arm in model_dict['arms']])  # gets all the time constants for the arms

    s = symbols('s')  # defines the laplace domain variable, s
    Q = Ee + sum((E_arms * T_arms * s) / (1 + T_arms * s))  # define the relaxance equation in the laplace domain
    Q_n = Q.normal()  # groups the relaxance equation into a single term and simplifies
    u_n = collect(expand(denom(Q_n)), s)  # simplifies the denominator and collects the coefficients of the polynomials
    q_n = collect(expand(numer(Q_n)), s)
    # returns in order n: 0, 1, 2, 3, ... N
    return array([u_n.coeff(s, i) for i in range(E_arms.size + 1)]).astype(float), \
           array([q_n.coeff(s, i) for i in range(E_arms.size + 1)]).astype(float)


# f_ts[i], u_matrix, q_matrix = maxwell_tip_sample_force_lee_and_radok(dt, ht[i], un, qn, u_matrix, q_matrix, R)
def maxwell_tip_sample_force_lee_and_radok(dt, h_i, un, qn, u_matrix, q_matrix, R):
    '''
    calculates the tip sample force between an AFM tip of a known radius and a viscoelastic, n arm, maxwell model
    :param dt: float simulation timestep
    :param h_i: float indentation of tip into sample at a given instance in time, i (h @ i)
    :param un: (n,) numpy array u coefficients for a maxwell model in the order u0, u1, u2, ... un (for n arms)
    :param qn: (n,) numpy array q coefficients for a maxwell model in the order q0, q1, q2, ... qn (for n arms)
    :param u_matrix: (2, n) numpy array containing history dependent stress derivatives according to the differential formulation of viscoelasticity
    :param q_matrix: (2, n) numpy array containing history dependent strain derivatives according to the differential formulation of viscoelasticity
    :param R: float radius of the AFM tip
    :return: force at the given instance in time, updated u_matrix, updated q_matrix
    '''
    # calculate the material response
    q_matrix[1, 0] = h_i**(3 / 2)  # lowest order strain derivative
    for j in range(qn.size - 1):  # higher order strain derivatives
        q_matrix[1, j + 1] = (q_matrix[1, j] - q_matrix[0, j]) / dt
    u_matrix[1, -1] = (16 * sqrt(R) / 3 * sum(qn * q_matrix[1]) - sum(un[: -1] * u_matrix[0, :-1])) / un[-1]  # highest order stress derivative
    for j in range(2, un.size + 1): # lower order stress derivatives
        u_matrix[1, -j] = u_matrix[0, -j] + u_matrix[1, -j + 1] * dt

    q_matrix[0] = q_matrix[1]  # save the current q state as the previous q state
    u_matrix[0] = u_matrix[1]  # save the current u state as the previous u state

    return u_matrix[1, 0], u_matrix, q_matrix  # force, updated u_matrix, updated q_matrix

