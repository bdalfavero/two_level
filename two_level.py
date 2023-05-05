#!/usr/bin/env python3

import sys
import pytomlpp as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylcp
#from pathos.pools import ProcessPool
import time

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

# Print a help message when the "--help" flag is passed.
if "--help" in sys.argv:
    print( \
    """
    two_level.py

    Simulation of two-level atom in a molasses.

    Command line interface:
    two_level.py <input file> <output file>
    Input is expected as TOML. Output is CSV.
    """
    )
    quit()

if len(sys.argv) < 3:
    print( \
    """
    Not enough arguments. Run "two_level.py --help" for more.
    """
    )
    quit()

# Read from the input file.
with open(sys.argv[1]) as fp:
    input_dict = pt.load(fp)

# Define constants.
mass = input_dict["experiment"]["mass"]
delta = input_dict["experiment"]["delta"]
s = input_dict["experiment"]["s"]
t_span = [0., input_dict["integration"]["t_final"]]

def magField(R):
    return np.zeros(R.shape)
laserBeams = return_lasers(delta, s)
hamiltonian = return_hamiltonian(0.)

# Make and solve the eqns of motion for N atoms.
n_atoms = input_dict["experiment"]["n_atoms"]
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

start_time = time.time()
# pool = ProcessPool(nodes=10)
# results = pool.map(solve_case, args_dict_list)
results = [solve_case(args_dict) for args_dict in args_dict_list]
end_time = time.time()

print("{} runs complete in {}s".format(n_atoms, end_time - start_time))

# Send output to csv file.
# Turn each solution individually into a DataFrame.
dfs = [pd.DataFrame( \
    np.concatenate((res.sol.v, res.sol.r), axis=0).T, \
    index=res.sol.t, columns=["vx", "vy", "vz", "x", "y", "z"]) \
    for res in results]

for df in dfs:
    df.index.name = "t"

# Assign a number to each solution.
for i in range(len(dfs)):
    dfs[i]["number"] = i

# Create a DataFrame containing each solution,
# indexed by the number.
df_total = pd.concat(dfs)

# Output the data as CSV.
df_total.to_csv(sys.argv[2])
