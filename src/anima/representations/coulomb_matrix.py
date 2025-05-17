import numpy as np

from ..utils import elements


class Coulomb_Matrix:
    """
    Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Müller, O. Anatole von Lilienfeld:
    Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning, Physical Review Letters 108(5): 058301, 2012. DOI 10.1103/PhysRevLett.108.058301
    """

    def __init__(self):
        pass

    def cmatrix(self, mol_in, upper=False):
        """
        Calculate the Coulomb Matrix of a sorted xyz molecule file from the funciton read_xyz.

        mol_in: structure read from a xyz file using the read_xyz function
        upper: return upper triangle matrix with diagonal if True
        """

        def distance(p1, p2):
            """
            Return the distance between two atoms locates at p1 and p2
            p1,p2 are atoms positions in cartesian coordinates
            """
            return np.linalg.norm(np.subtract(p2, p1))

        atoms = np.array(mol_in["atom"], dtype="str")
        Natoms = len(atoms)
        pos = []
        for i in range(Natoms):
            pos.append(np.array(mol_in.iloc[i][1:4], dtype="float"))
        pos = np.array(pos)

        # distances
        cmat = np.zeros((Natoms, Natoms))
        for j in range(Natoms):
            for i in range(j):
                zi = elements(mol_in["atom"][i], "Z")
                zj = elements(mol_in["atom"][j], "Z")
                cmat[i, j] = zi * zj / distance(pos[i], pos[j])  # type: ignore

        # diagonal
        for i in range(Natoms):
            cmat[i, i] = 0.5 * (elements(mol_in["atom"][i], "Z")) ** (2.4)  # type: ignore

        if upper is not True:
            for j in range(Natoms):
                for i in range(j):
                    cmat[j, i] = cmat[i, j]
        return cmat
