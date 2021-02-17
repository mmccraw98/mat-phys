from sympy import symbols, denom, expand, collect, numer
from numpy import array, zeros


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



