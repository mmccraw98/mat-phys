from sympy import symbols, denom, expand, collect, numer
from numpy import array, zeros, sqrt, ones


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


def f_ts_jeff_williams_gen_maxwell_model(i, zt, x, xd, vd, force, G, eta, R, dt, A=0, a0=2e-10):
    '''
    calculates the tip sample force for the generalized, n arm, maxwell model according to the formulation presented
    by jeff williams in his thesis for a single arm (SLS) maxwell
    :param i: int instance in time
    :param zt: (length time) numpy array tip position
    :param x: (length time) numpy array position of the sample surface
    :param xd: (n, length time) numpy array position of the dampers in the surface
    :param vd: (n, length time) numpy array velocities of the dampers in the surface
    :param force: (length time) numpy array force between the tip and the sample
    :param G: (n + 1) numpy array elastic moduli of the springs in the model (G[0] is the elastic term)
    :param eta: (n) numpy array viscosities of the dampers in the model
    :param R: float radius of tip
    :param dt: float timestep of simulation
    :param A: float hamaker constant for the tip-sample adhesion interaction (default as 0 to have no adhesion)
    :param a0: float interatomic spacing (default as 2 angstroms)
    :return: updated material values and tip-sample force at time i+1: force, x, xd, vd
    '''
    # contact
    if zt[i] < x[i] + a0:
        # prescribe the compression of the surface due to the tip position
        x[i + 1] = zt[i + 1] - a0
        # calculate the forces in the arms due to their springs
        f_arms = - G[1:] * (x[i + 1] - xd[:, i])
        # calculate the velocities of the dampers due to the forces in the springs
        vd[:, i + 1] = - f_arms / eta
        # calculate the positions of the dampers from the velocity update
        xd[:, i + 1] = xd[:, i] + vd[:, i + 1] * dt
        # ensure that the dampers do not exceed the position of the springs
        #xd[:, i + 1] = xd[:, i + 1] * (xd[:, i + 1] <= x[i + 1]) + x[i + 1] * (xd[:, i + 1] > x[i + 1])
        # calculate sum of forces: elastic spring force, arm forces, vdw force
        force[i + 1] = - G[0] * x[i + 1] + sum(f_arms)# - A * R / (6 * a0 ** 2)
    # non-contact
    else:
        # the model is either compressed or not.  if it is not compressed, nothing happens
        # if it is compressed, the force stored in the elastic spring will just dissipate
        # and cause the rest of the model to return to normal position -> elastic spring force is
        # the input, sample position is the output
        #f_vdw = - A * R / (6 * (zt[i] - x[i]) ** 2)
        #if f_vdw < - A * R / (6 * a0 ** 2):
        #    f_vdw = - A * R / (6 * a0 ** 2)
        # calculate the force acting on the elastic arm (spring force and vdw)
        f_e = - G[0] * x[i]# - f_vdw
        # the elastic spring then pulls on the arms (force in the arms is the opposite of the elastic arm force)
        # calculate the velocity in the dampers from the force in the arms
        vd[:, i + 1] = f_e / eta
        # calculate the position of the dampers from their velocities and previous positions
        xd[:, i + 1] = xd[:, i] + vd[:, i + 1] * dt
        # calculate the new position of the sample, derivation in jupyter notebook
        x[i + 1] = 1 / sum(G) * sum(G[1:] * (xd[:, i] + vd[:, i + 1] * dt))
        # the sample and tip are pulled towards each other from adhesion
        force[i + 1] = 0#f_vdw
    return force, x, xd, vd