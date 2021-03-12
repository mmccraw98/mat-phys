from numpy.random import rand, randint
from numpy import inf, sin, cos, pi, arange, sum, tile, max, min, zeros, ones, array, where, abs, sqrt, arctan, tan, cumsum, real, imag
from auxilary import fastmovingaverage, tic, toc, logistic

class Sample:

    def __init__(self, youngs_modulus, size, surface_map):
        '''
        :param youngs_modulus: (int/float) Elastic Modulus of the Sample
        :param size: (3x1 or 2x1 tuple of int/float nm value) Dimensions of the Sample, either 2 with
                     an implied infinite thickness or 3 with a specified thickness as the last dimension.
                     In either case, the thickness value will be taken from the mean value of the
                     surface topography map
        :param surface_map: (int or 2-D numpy matrix) Can be a 2-D numpy matrix which
        '''
        self.youngs_modulus = youngs_modulus
        if len(size) < 3:
            self.size = (size[0], size[1], inf)
        else:
            self.size = size

        # randomly generate a surface_map
        dimension_length = int(10e3)
        resolution = (size[0]/dimension_length, size[1]/dimension_length, size[2]/dimension_length)
        if surface_map.lower() in ['r', 'rand', 'random', 'rd', 'rndm']:
            harmonics_x = sum([sin(rand() * 2*pi * arange(0, self.size[0], resolution[0])) +
                               cos(rand() * 2*pi * arange(0, self.size[0], resolution[0]))
                               for i in range(randint(1, 2))], axis=0)
            harmonics_y = sum([sin(rand() * 2*pi * arange(0, self.size[1], resolution[1])) +
                               cos(rand() * 2*pi * arange(0, self.size[0], resolution[1]))
                               for i in range(randint(1, 2))], axis=0)
            self.surface_map = tile(harmonics_x, (dimension_length, 1)) * tile(harmonics_y, (dimension_length, 1)).T / 10

class elasticSample:
    def __init__(self, youngs_modulus, poissons, surface_E, max_harmonics=3):
        self.youngs_modulus = youngs_modulus
        self.poissons = poissons
        self.surface_E = surface_E
        self.frequencies_x = [rand()/10 for i in range(randint(1, max_harmonics))]
        self.frequencies_y = [rand()/10 for i in range(randint(1, max_harmonics))]
    def get_height(self, xy_position):
        harmonics_x = sum([sin(f * 2*pi * xy_position[0]) + cos(f * 2*pi * xy_position[0])
                           for f in self.frequencies_x], axis=0)
        harmonics_y = sum([sin(f * 2*pi * xy_position[1]) + cos(f * 2*pi * xy_position[1])
                           for f in self.frequencies_y], axis=0)
        return harmonics_x * harmonics_y / 10

class AFM:
    def __init__(self, R, E, b, h, L, p, Q, state_base, time_domain, state_tip=None):
        self.R = R
        self.E = E
        self.k = 3 * (b * h**3 / 12) * E / L
        self.p = p
        self.w0 = sqrt(self.k / (p / b * h * L))
        self.Q = Q
        self.state_base = state_base  # [x pos, y pos, z pos] -> typically functions
        if state_tip is not None:  # define cantilever to start at a specified position
            self.state_tip = state_tip  # [z accel, z vel, z pos] -> other positions obtained from the base
        else:  # define cantilever to start at rest, at a height even with the base
            self.state_tip = array([0, 0, self.state_base[2][0]])
        self.time_domain = time_domain

    def eom(self, state_tip, state_base, interaction_force=0):
        return (-self.k/(self.w0*self.Q)*state_tip[1] - self.k*(state_tip[2] - state_base[2])
                + interaction_force) * self.w0**2 / self.k

    def elastic_repulsion(self, state_tip, state_base, surface):
        state_surface = surface.get_height(state_base[: 2])
        return surface.youngs_modulus * (state_surface - state_tip[2]) * (state_tip[2] < state_surface)

    def lennard_jones(self, state_tip, state_base, surface):
        r = surface.get_height(state_base[: 2]) - state_tip

    def elastic_hertz_rigid(self, state_tip, state_base, surface):  # no deformation of probe
        state_surface = surface.get_height(state_base[: 2])
        E_st = ((1-surface.poissons**2) / surface.youngs_modulus) ** -1
        d = (state_surface - state_tip[2]) * (state_tip[2] < state_surface)
        return (4/3) * E_st * self.R**(1/2) * (d) ** (3/2)


    # need to use a better way of doing this
    # ideally we define an eom initially with the state_tip and base as well as interaction_force
    # then we call verlet_solver without having to re-pass eom again
    def verlet_solver(self, surface):
        dt = self.time_domain[1] - self.time_domain[0]
        base_log = zeros(self.time_domain.shape)
        tip_log = zeros(self.time_domain.shape)
        sample_log = zeros(self.time_domain.shape)
        force_log = zeros(self.time_domain.shape)
        state_tip_next = zeros(self.state_tip.shape)
        for i in range(self.time_domain.shape[0]):
            interaction = 0#self.elastic_hertz_rigid(self.state_tip, self.state_base[:, i], surface)
            state_tip_next[2] = self.state_tip[2] + self.state_tip[1] * dt + (1/2) * self.state_tip[0] * dt**2
            state_tip_next[0] = self.eom(self.state_tip, self.state_base[:, i], interaction)
            state_tip_next[1] = self.state_tip[1] + (1/2) * (self.state_tip[0] + state_tip_next[0]) * dt
            force_log[i] = interaction
            base_log[i] = self.state_base[2][i]
            tip_log[i] = self.state_tip[2]
            sample_log[i] = surface.get_height(self.state_base[:, i][: 2])
            self.state_tip = state_tip_next
        return base_log, tip_log, sample_log, force_log

import matplotlib.pyplot as plt
sample = elasticSample(youngs_modulus=1000, poissons=0.3, surface_E=1)

dt = 0.0001
T = 200
time = arange(0, T+dt, dt)
base_x, base_y = zeros(time.shape), zeros(time.shape)
excitation_f = 100*sum([logistic(time, s=20*i, k=0.5) for i in arange(1, 11, 1)], axis=0)
base_z = sin(excitation_f*time)
weak = AFM(R=1, E=1e6, b=0.1, h=0.1, L=1, p=1, Q=1, state_base=array([base_x, base_y, base_z]), time_domain=time)
base, tip, sample, force = weak.verlet_solver(sample)
plt.plot(time, tip, label='tip')
plt.plot(time, base, label='base')
plt.grid()
plt.legend()
plt.show()