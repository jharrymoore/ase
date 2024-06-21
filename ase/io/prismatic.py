"""Module to read and write atoms in xtl file format for the prismatic and
computem software.

See https://prism-em.com/docs-inputs for an example of this format and the
documentation of prismatic.

See https://sourceforge.net/projects/computem/ for the source code of the
computem software.
"""

import numpy as np

from ase.atoms import Atoms, symbols2numbers
from ase.utils import reader
from .utils import verify_cell_for_export, verify_dictionary


@reader
def read_prismatic(fd):
    r"""Import prismatic and computem xyz input file as an Atoms object.

    Reads cell, atom positions, occupancies and Debye Waller factor.
    The occupancy values and the Debye Waller factors are obtained using the
    `get_array` method and the `occupancies` and `debye_waller_factors` keys,
    respectively. The root means square (RMS) values from the
    prismatic/computem xyz file are converted to Debye-Waller factors (B) in Å²
    by:

    .. math::

        B = RMS^2 * 8\pi^2

    """
    # Read comment:
    fd.readline()

    # Read unit cell parameters:
    cellpar = [float(i) for i in fd.readline().split()]

    # Read all data at once
    # Use genfromtxt instead of loadtxt to skip last line
    read_data = np.genfromtxt(fname=fd, skip_footer=1)
    # Convert from RMS to Debye-Waller factor
    RMS = read_data[:, 5] ** 2 * 8 * np.pi**2

    atoms = Atoms(
        symbols=read_data[:, 0],
        positions=read_data[:, 1:4],
        cell=cellpar,
    )
    atoms.set_array("occupancies", read_data[:, 4])
    atoms.set_array("debye_waller_factors", RMS)

    return atoms


class XYZPrismaticWriter:
    """See the docstring of the `write_prismatic` function."""

    def __init__(self, atoms, debye_waller_factors=None, comments=None):
        verify_cell_for_export(atoms.get_cell())

        self.atoms = atoms.copy()
        self.atom_types = set(atoms.symbols)
        self.comments = comments

        self.occupancies = self._get_occupancies()
        debye_waller_factors = self._get_debye_waller_factors(debye_waller_factors)
        self.RMS = np.sqrt(debye_waller_factors / (8 * np.pi**2))

    def _get_occupancies(self):
        if "occupancies" in self.atoms.arrays:
            occupancies = self.atoms.get_array("occupancies", copy=False)
        else:
            occupancies = np.ones_like(self.atoms.numbers)

        return occupancies

    def _get_debye_waller_factors(self, DW):
        if np.isscalar(DW):
            if len(self.atom_types) > 1:
                raise ValueError(
                    "This cell contains more then one type of "
                    "atoms and the Debye-Waller factor needs to "
                    "be provided for each atom using a "
                    "dictionary."
                )
            DW = np.ones_like(self.atoms.numbers) * DW
        elif isinstance(DW, dict):
            verify_dictionary(self.atoms, DW, "DW")
            # Get the arrays of DW from mapping the DW defined by symbol
            DW = {symbols2numbers(k)[0]: v for k, v in DW.items()}
            DW = np.vectorize(DW.get)(self.atoms.numbers)
        else:
            for name in ["DW", "debye_waller_factors"]:
                if name in self.atoms.arrays:
                    DW = self.atoms.get_array(name)

        if DW is None:
            raise ValueError(
                "Missing Debye-Waller factors. It can be "
                "provided as a dictionary with symbols as key or "
                "can be set for each atom by using the "
                '`set_array("debye_waller_factors", values)` of '
                "the `Atoms` object."
            )

        return DW

    def _get_file_header(self):
        # 1st line: comment line
        if self.comments is None:
            s = "{0} atoms with chemical formula: {1}.".format(
                len(self.atoms), self.atoms.get_chemical_formula()
            )
        else:
            s = self.comments

        s = s.strip()
        s += " generated by the ase library.\n"
        # 2nd line: lattice parameter
        s += "{} {} {}".format(*self.atoms.cell.cellpar()[:3])

        return s

    def write_to_file(self, f):
        data_array = np.vstack(
            (self.atoms.numbers, self.atoms.positions.T, self.occupancies, self.RMS)
        ).T

        np.savetxt(
            fname=f,
            X=data_array,
            fmt="%.6g",
            header=self._get_file_header(),
            newline="\n",
            footer="-1",
            comments="",
        )


def write_prismatic(fd, *args, **kwargs):
    r"""Write xyz input file for the prismatic and computem software. The cell
    needs to be orthorhombric. If the cell contains the `occupancies` and
    `debye_waller_factors` arrays (see the `set_array` method to set them),
    these array will be written to the file.
    If the occupancies is not specified, the default value will be set to 0.

    Parameters:

    atoms: Atoms object

    debye_waller_factors: float or dictionary of float or None (optional).
        If float, set this value to all
        atoms. If dictionary, each atom needs to specified with the symbol as
        key and the corresponding Debye-Waller factor as value.
        If None, the `debye_waller_factors` array of the Atoms object needs to
        be set (see the `set_array` method to set them), otherwise raise a
        ValueError. Since the prismatic/computem software use root means square
        (RMS) displacements, the Debye-Waller factors (B) needs to be provided
        in Å² and these values are converted to RMS displacement by:

        .. math::

            RMS = \sqrt{\frac{B}{8\pi^2}}

        Default is None.

    comment: str (optional)
        Comments to be written in the first line of the file. If not
        provided, write the total number of atoms and the chemical formula.

    """

    writer = XYZPrismaticWriter(*args, **kwargs)
    writer.write_to_file(fd)
