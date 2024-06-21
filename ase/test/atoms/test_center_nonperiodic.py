def test_center_nonperiodic():
    import numpy as np
    from ase import Atoms

    a = Atoms("H")
    a.center(about=[0.0, 0.0, 0.0])
    print(a.cell)
    print(a.positions)

    assert not a.cell.any()
    assert not a.positions.any()

    a.cell = [0.0, 2.0, 0.0]
    a.center()
    print(a)
    print(a.positions)
    assert np.abs(a.positions - [[0.0, 1.0, 0.0]]).max() < 1e-15

    a.center(about=[0.0, -1.0, 1.0])
    print(a.positions)
    assert np.abs(a.positions - [[0.0, -1.0, 1.0]]).max() < 1e-15
    assert np.abs(a.cell - np.diag([0.0, 2.0, 0.0])).max() < 1e-15
    a.center(axis=2, vacuum=2.0)
    print(a.positions)
    print(a.cell)
    assert np.abs(a.positions - [[0.0, -1.0, 2.0]]).max() < 1e-15
    assert np.abs(a.cell - np.diag([0.0, 2.0, 4.0])).max() < 1e-15
