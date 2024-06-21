# flake8: noqa
import numpy as np
from ase import Atoms
from ase.calculators.gulp import GULP, Conditions

cluster = Atoms(
    symbols="O4SiOSiO2SiO2SiO2SiOSiO2SiO3SiO3H8",
    pbc=np.array([False, False, False], dtype=bool),
    cell=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    positions=np.array(
        [
            [-1.444348, -0.43209, -2.054785],
            [-0.236947, 2.98731, 1.200025],
            [3.060238, -1.05911, 0.579909],
            [2.958277, -3.289076, 2.027579],
            [-0.522747, 0.847624, -2.47521],
            [-2.830486, -2.7236, -2.020633],
            [-0.764328, -1.251141, 1.402431],
            [3.334801, 0.041643, -4.168601],
            [-1.35204, -2.009562, 0.075892],
            [-1.454655, -1.985635, -1.554533],
            [0.85504, 0.298129, -3.159972],
            [1.75833, 1.256026, 0.690171],
            [2.376446, -0.239522, -2.881245],
            [1.806515, -4.484208, -2.686456],
            [-0.144193, -2.74503, -2.177778],
            [0.167583, 1.582976, 0.47998],
            [-1.30716, 1.796853, -3.542121],
            [1.441364, -3.072993, -1.958788],
            [-1.694171, -1.558913, 2.704219],
            [4.417516, 1.263796, 0.563573],
            [3.066366, 0.49743, 0.071898],
            [-0.704497, 0.351869, 1.102318],
            [2.958884, 0.51505, -1.556651],
            [1.73983, -3.161794, -0.356577],
            [2.131519, -2.336982, 0.996026],
            [0.752313, -1.788039, 1.687183],
            [-0.142347, 1.685301, -1.12086],
            [2.32407, -1.845905, -2.588202],
            [-2.571557, -1.937877, 2.604727],
            [2.556369, -4.551103, -3.2836],
            [3.032586, 0.591698, -4.896276],
            [-1.67818, 2.640745, -3.27092],
            [5.145483, 0.775188, 0.95687],
            [-2.81059, -3.4492, -2.650319],
            [2.558023, -3.594544, 2.845928],
            [0.400993, 3.469148, 1.733289],
        ]
    ),
)


c = Conditions(cluster)
c.min_distance_rule("O", "H", ifcloselabel1="O2", ifcloselabel2="H", elselabel1="O1")
calc = GULP(keywords="conp", shel=["O1", "O2"], conditions=c)

# Single point calculation
cluster.calc = calc
print(cluster.get_potential_energy())

# Optimization using the internal optimizer of GULP
calc.set(keywords="conp opti")
opt = calc.get_optimizer(cluster)
opt.run(fmax=0.05)
print(cluster.get_potential_energy())
