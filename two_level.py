#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pylcp
from pathos.pools import ProcessPool

# This simulation is based on the 1D optical molasses in pyLCP's documention.


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

def solve_case(args_dict):
    """
    Solve for the motion of a single atom.
    args_dict = {
        "laserBeams": ...,
        "magField": ...,
        "hamiltonian": ...,
        "t_span": ...,
        "r0": ...,
        "v0": ...,
    }
    """

    eqns = pylcp.obe(args_dict["laserBeams"], 
            args_dict["magField"], args_dict["hamiltonian"])
    eqns.set_initial_position_and_velocity(args_dict["r0"], \
                                            args_dict["v0"])
    eqns.set_initial_rho_from_rateeq()
    eqns.evolve_motion(t_span,
                      random_recoil=True, progress_bar=False,
                      max_scatter_probability=0.25,
                      freeze_axis=[False, True, True])
    return eqns

# Define constants.
mass = 200
delta = -2.
s = 1.5
t_span = [0., 10*mass*(1+2*s+4*np.abs(delta)**2)/s]

# Define Hamiltonian and EM fields.
def magField(R):
    return np.zeros(R.shape)
laserBeams = return_lasers(delta, s)
hamiltonian = return_hamiltonian(0.)

# Make and solve the eqns of motion for N atoms.
n_atoms = 2
args_dict_list = []
for i in range(n_atoms):
    args_dict_list.append({
        "laserBeams": laserBeams,
        "magField": magField,
        "hamiltonian": hamiltonian,
        "tspan": t_span,
        "r0": np.array([0., 0., 0.]),
        "v0": np.array([0., 0., 0.])
    })

pool = ProcessPool(nodes=5)
results = pool.map(solve_case, args_dict_list)

eqns_final = results[0]

fig, ax = plt.subplots()
ax.plot(eqns_final.sol.t, eqns_final.sol.v.T)
plt.show()
