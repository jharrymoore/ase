import pytest
import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.io import read, write

# Check that saving/loading pdb files correctly reproduces the atoms object.
#
# Loading will restore the cell from lengths/angles, so the best we can do
# is to recreate the scaled positions, not the absolute positions.

images = [
    Atoms(
        symbols="C8O8Ru64",
        pbc=np.array([True, True, True], dtype=bool),
        cell=np.array(
            [
                [9.46101634e00, 5.46231901e00, -7.62683750e-07],
                [0.00000000e00, 1.09246400e01, -7.62683750e-07],
                [0.00000000e00, 0.00000000e00, 2.14654300e01],
            ]
        ),
        positions=np.array(
            [
                [7.80131882e-01, 6.83747136e00, 8.38204657e00],
                [5.51092271e00, 9.56854231e00, 8.38223568e00],
                [3.07270715e00, 2.87955302e00, 8.37140640e00],
                [7.80350536e00, 5.61092571e00, 8.37131213e00],
                [4.27438360e00, 6.25264571e00, 7.97264597e00],
                [9.00498640e00, 8.98355980e00, 7.97286435e00],
                [6.37288973e00, 1.35474697e01, 7.83982205e00],
                [1.64255024e00, 1.08160090e01, 7.83994023e00],
                [7.71235875e-01, 6.84831577e00, 9.54503003e00],
                [5.50223518e00, 9.57934418e00, 9.54476437e00],
                [3.03497100e00, 2.94960249e00, 9.52997683e00],
                [7.76620880e00, 5.68120179e00, 9.52999700e00],
                [4.23008702e00, 6.32508468e00, 9.15933250e00],
                [8.96060688e00, 9.05590199e00, 9.15923442e00],
                [6.34874076e00, 1.35969943e01, 9.03839912e00],
                [1.61820848e00, 1.08649253e01, 9.01849841e00],
                [1.57683637e00, 2.73116147e00, -2.54228044e-07],
                [1.56720630e00, 2.72722886e00, 4.28570884e00],
                [7.88417713e-01, 1.36558046e00, 2.15514407e00],
                [8.02210750e-01, 1.34385101e00, 6.43536380e00],
                [3.94209046e00, 4.09674123e00, -4.44898981e-07],
                [3.95116212e00, 4.10376637e00, 4.28640956e00],
                [3.15367180e00, 2.73116022e00, 2.15514388e00],
                [3.15302826e00, 2.73391087e00, 6.47587998e00],
                [6.30734454e00, 5.46232098e00, -6.35569919e-07],
                [6.29772257e00, 5.45840160e00, 4.28564811e00],
                [5.51892683e00, 4.09674051e00, 2.15514369e00],
                [5.53267073e00, 4.07509705e00, 6.43527963e00],
                [8.67259863e00, 6.82790073e00, -8.26240856e-07],
                [8.68166544e00, 6.83494012e00, 4.28642358e00],
                [7.88417997e00, 5.46231972e00, 2.15514350e00],
                [7.88362942e00, 5.46507502e00, 6.47590212e00],
                [1.57683637e00, 5.46232147e00, -4.44898981e-07],
                [1.58727500e00, 5.44486645e00, 4.24854361e00],
                [7.88417713e-01, 4.09674046e00, 2.15514388e00],
                [8.01608482e-01, 4.07705367e00, 6.44578990e00],
                [3.94209046e00, 6.82790123e00, -6.35569919e-07],
                [3.95243122e00, 6.81073405e00, 4.32065689e00],
                [3.15367180e00, 5.46232022e00, 2.15514369e00],
                [3.16456215e00, 5.44150374e00, 6.44566316e00],
                [6.30734454e00, 8.19348098e00, -8.26240856e-07],
                [6.31780039e00, 8.17600912e00, 4.24852811e00],
                [5.51892588e00, 6.82789997e00, 2.15514350e00],
                [5.53216899e00, 6.80824683e00, 6.44574538e00],
                [8.67259863e00, 9.55906073e00, -1.01691179e-06],
                [8.68296348e00, 9.54187626e00, 4.32068356e00],
                [7.88417997e00, 8.19347972e00, 2.15514331e00],
                [7.89512792e00, 8.17267187e00, 6.44565547e00],
                [1.57683637e00, 8.19348147e00, -6.35569919e-07],
                [1.58115689e00, 8.20071292e00, 4.29055653e00],
                [7.88417713e-01, 6.82790046e00, 2.15514369e00],
                [7.91948666e-01, 6.82222698e00, 6.48549188e00],
                [3.94209046e00, 9.55906123e00, -8.26240856e-07],
                [3.93358820e00, 9.55894698e00, 4.29187459e00],
                [3.15367180e00, 8.19348022e00, 2.15514350e00],
                [3.15825664e00, 8.18574447e00, 6.38108109e00],
                [6.30734454e00, 1.09246410e01, -1.01691179e-06],
                [6.31166355e00, 1.09318806e01, 4.29057142e00],
                [5.51892588e00, 9.55905997e00, 2.15514331e00],
                [5.52249944e00, 9.55339051e00, 6.48545486e00],
                [8.67259863e00, 1.22902207e01, -1.20758273e-06],
                [8.66410508e00, 1.22901152e01, 4.29183559e00],
                [7.88418091e00, 1.09246403e01, 2.15514312e00],
                [7.88880125e00, 1.09169018e01, 6.38105940e00],
                [1.57683637e00, 1.09246415e01, -8.26240856e-07],
                [1.58687157e00, 1.09077863e01, 4.32193338e00],
                [7.88417713e-01, 9.55906046e00, 2.15514350e00],
                [7.78394031e-01, 9.52397919e00, 6.44162756e00],
                [3.94209046e00, 1.22902212e01, -1.01691179e-06],
                [3.95166143e00, 1.22749617e01, 4.26568019e00],
                [3.15367180e00, 1.09246402e01, 2.15514331e00],
                [3.19235336e00, 1.09175437e01, 6.44091634e00],
                [6.30734454e00, 1.36558010e01, -1.20758273e-06],
                [6.31737307e00, 1.36389093e01, 4.32189961e00],
                [5.51892588e00, 1.22902200e01, 2.15514312e00],
                [5.50895045e00, 1.22551312e01, 6.44172685e00],
                [8.67259863e00, 1.50213807e01, -1.39825367e-06],
                [8.68213569e00, 1.50061426e01, 4.26566744e00],
                [7.88418091e00, 1.36558003e01, 2.15514293e00],
                [7.92283529e00, 1.36487476e01, 6.44087611e00],
            ]
        ),
    ),
]


@pytest.mark.parametrize("nrepeat", [1, 2])
def test_pdb_cell_io(nrepeat):
    traj1 = images * nrepeat
    write("grumbles.pdb", traj1)
    traj2 = read("grumbles.pdb", index=":")

    assert len(traj1) == len(traj2)
    for atoms1, atoms2 in zip(traj1, traj2):
        spos1 = (atoms1.get_scaled_positions() + 0.5) % 1.0
        spos2 = (atoms2.get_scaled_positions() + 0.5) % 1.0
        cell1 = atoms1.cell.cellpar()
        cell2 = atoms2.cell.cellpar()

        np.testing.assert_allclose(
            atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers()
        )
        np.testing.assert_allclose(spos1, spos2, rtol=0, atol=2e-4)
        np.testing.assert_allclose(cell1, cell2, rtol=0, atol=1e-3)


def test_pdb_nonbulk_read():
    atoms1 = fcc111("Au", size=(3, 3, 1))
    atoms1.symbols[4:10] = "Ag"
    atoms1.write("test.pdb")
    atoms2 = read("test.pdb")

    spos1 = (atoms1.get_scaled_positions() + 0.5) % 1.0
    spos2 = (atoms2.get_scaled_positions() + 0.5) % 1.0

    np.testing.assert_allclose(atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers())
    np.testing.assert_allclose(spos1, spos2, rtol=0, atol=2e-4)


def test_pdb_no_periodic():
    atoms1 = Atoms("H")
    atoms1.center(vacuum=1)
    atoms1.write("h.pdb")

    atoms2 = read("h.pdb")

    spos1 = atoms1.get_positions()
    spos2 = atoms2.get_positions()

    np.testing.assert_allclose(atoms1.get_atomic_numbers(), atoms2.get_atomic_numbers())
    np.testing.assert_allclose(spos1, spos2, rtol=0, atol=2e-4)
