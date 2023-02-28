#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pylcp

# This simulation is based on the 1D optical molasses in pyLCP's documention.

mass = 200
delta = -2.
s = 1.5

# Make a method to return the lasers:
def return_lasers(delta, s):
    return pylcp.laserBeams([
        {'kvec':np.array([1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
         'pol_coord':'spherical', 'delta':delta, 's':s},
        {'kvec':np.array([-1., 0., 0.]), 'pol':np.array([0., 1., 0.]),
         'pol_coord':'spherical', 'delta':delta, 's':s},
        ], beam_type=pylcp.infinitePlaneWaveBeam)

# Now define a two level Hamiltonian, connected using pi-light.
def return_hamiltonian(delta):
    Hg = np.array([[0.]])
    He = np.array([[-delta]])
    mu_q = np.zeros((3, 1, 1))
    d_q = np.zeros((3, 1, 1))
    d_q[1, 0, 0] = 1.

    return pylcp.hamiltonian(Hg, He, mu_q, mu_q, d_q, mass=mass)

# Define Hamiltonian and EM fields.
magField = lambda R: np.zeros(R.shape)
laserBeams = return_lasers(delta, s)
hamiltonian = return_hamiltonian(0.)

# Make and solve the eqns of motion.
eqns = pylcp.obe(laserBeams, magField, hamiltonian)
eqns.set_initial_position_and_velocity(np.array([0., 0., 0.]), np.array([0., 0., 0.]))
eqns.set_initial_rho_from_rateeq()
eqns.evolve_motion([0., 10*mass*(1+2*s+4*np.abs(delta)**2)/s],
                      random_recoil=True, progress_bar=True,
                      max_scatter_probability=0.25,
                      freeze_axis=[False, True, True])

# Plot the velocity vs time.
fig, ax = plt.subplots()
ax.plot(eqns.sol.t, eqns.sol.v.T)
plt.show()
