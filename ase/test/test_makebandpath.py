from ase.dft.kpoints import bandpath
from ase.build import bulk


def test_makebandpath():
    atoms = bulk("Au")
    cell = atoms.cell

    path0 = bandpath("GXL", cell)
    print(path0)
    path1 = bandpath([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell, npoints=50)
    print(path1)
    path2 = bandpath(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.1, 0.2, 0.3]],
        cell,
        npoints=50,
        special_points={"G": [0.0, 0.0, 0.0]},
    )
    print(path2)
