"""
A collection of tools to manipulate SMILES strings

Created by © Rodrigo Carvalho 2020
Maintained by © Rodrigo Carvalho
"""

import os
from typing import List, Tuple

import numpy as np

## for openbabel < 3
import openbabel.pybel as pybel
import pysmiles
from joblib import Parallel, delayed
from pysmiles import read_smiles
from rdkit import Chem

from .lib import xyz2mol
from .utils import elements

# for openbabel 2.4
# import pybel


class SMILES:
    def smiles_cleaner(self, s: str) -> str:
        """Returns a 'clean' version of the SMILES to be used in
        Language Processing.

        Parameters
        ----------
        s : string
            SMILES

        Returns
        -------
        string
            SMILES clean version

        """
        return (
            s.replace("[C@H]", "C")
            .replace("[C@@H]", "C")
            .replace("/", "")
            .replace("\\", "")
            .replace(".", "")
            .replace("[Br]", "Br")
            .replace("[C@@]", "C")
            .replace("[C@]", "C")
            .replace("[CH2]", "C")
            .replace("[CH]", "C")
            .replace("[C]", "C")
            .replace("[H]", "")
            .replace("[N@H]", "N")
            .replace("[N@@H]", "N")
            .replace("[NH2]", "N")
            .replace("[NH]", "N")
            .replace("[N]", "N")
            .replace("[O]", "O")
            .replace("[S@@]", "S")
            .replace("[S@]", "S")
            .replace("[S]", "S")
            .replace("[c]", "c")
            .replace("[n]", "n")
            .replace("[N@]", "N")
            .replace("[N@@]", "N")
            .replace("[F]", "F")
            .replace("[Cl]", "Cl")
        )

    def OB_xyz_to_smiles(self, fname: str) -> str:
        """Read a xyz file and convert it into a SMILES stringself.
        ## Based on OpenBabel, thus requires OpenBabel and pybel

        Parameters
        ----------
        fname : str
            file name/path

        Returns
        -------
        str
            SMILES

        """
        mol = next(pybel.readfile("xyz", fname))

        smi = mol.write(format="smi")

        return str(smi).split()[0].strip()

    def OB_standard_smiles(self, s: str, kekule: bool = False) -> str:
        """Read a SMILES string and convert it into a Canonical SMILES.
        ## Based on OpenBabel, thus requires OpenBabel/pybel

        Parameters
        ----------
        fname : str
            SMILES

        Returns
        -------
        str
            SMILES

        """
        mol = pybel.readstring("smi", s)

        if kekule:
            smi = mol.write(format="can", opt={"k": None})
        else:
            smi = mol.write(format="can")

        return str(smi).split()[0]

    def PS_fix(self, s: str) -> str:
        """Read a SMILES string and fix some inconsistencies
        based on PySMILES
        Parameters
        ----------
        fname : str
            SMILES

        Returns
        -------
        str
            SMILES

        """
        return pysmiles.write_smiles(pysmiles.read_smiles(s))

    def xyz_to_smiles(
        self, fname: str, chiral: bool = False, charged_fragments: bool = False
    ) -> str:
        """Read a xyz file and convert it into a canonical SMILES string.
        ## requires xyz2mol.py (https://github.com/jensengroup/xyz2mol)

        Parameters
        ----------
        fname : type
            file name/path
        kekulize : bool
            return a kekulized SMILES if True
        can : bool
            return a canonical SMILES if True

        Returns
        -------
        string
            SMILES

        """
        phrase = "python " + xyz2mol.__file__ + " " + str(fname)
        if not chiral:
            phrase += " --ignore-chiral"
        if not charged_fragments:
            phrase += " --no-charged-fragments"

        return os.popen(phrase).read().splitlines()[0]
        # from .lib import xyz2mol
        # modd = xyz2mol
        # atoms, charge, coordinates = modd.read_xyz_file(fname)
        # mol = modd.xyz2mol(atoms, coordinates, charge, allow_charged_fragments=True, embed_chiral=False)
        # if kekulize == True: modd.Chem.Kekulize(mol)
        # return modd.Chem.MolToSmiles(mol, isomericSmiles=False, canonical=can)

    def standard_smiles(self, s: str, kekule: bool = False, can: bool = True) -> str:
        """Return the SMILES in a standad form to be used in Language
        Processing adopting the Kekule and/or Canonical forms.

        Parameters
        ----------
        s : string
            SMILES

        Returns
        -------
        string
            SMILES

        """
        m = Chem.MolFromSmiles(s)
        if kekule:
            Chem.Kekulize(m)
        return Chem.MolToSmiles(
            m, isomericSmiles=False, kekuleSmiles=kekule, canonical=can
        )

    def get_hydrogens(self, fname: str) -> int:
        """Function to return the number of hydrogens from a xyz file.

        Parameters
        ----------
        fname : string/path
            path to xyz file

        Returns
        -------
        int
            Number of hydrogens

        """
        with open(fname, mode="r") as f:
            bb = f.read().splitlines()
            n_of_h = 0
            for i in bb:
                if "H" in i:
                    n_of_h += 1
        return n_of_h

    def smilesSEP(self, s: str, pack_bonds: bool = False) -> List[str]:
        """Process a SMILES string and return its elements
        separated in a list.

        - In this version all the [] segments are considered an entire element
        - Brackets () and '/' or '\' are considered elements
        - Numbers are common elements
        - explicity bonds are packed based on pack_bonds arg

        Parameters
        ----------
        s : string
            SMILES
        pack_bonds : bool
            if True, explicity bonds from the SMILES will be packed
            with the next element of the sequence. Ex.: C=C will result in
            [C, =C] instead of [C, =, C]

        Returns
        -------
        list
            list of the separated elements from the SMILES.

        """
        aromaticity = ["c", "s", "se", "o", "b", "n", "p"]
        bonds = ["#", "=", "$"]
        sbackets_start = ["["]
        sbackets_finish = ["]"]
        sep = []  # list of separated elements
        brac_square = ""
        key_square = 0  # key to check if its a square bracket
        count = 0
        flag0 = False
        while count < len(s):
            i = s[count]
            if key_square != 0:  # check if the pointer is inside a bracket
                if not np.isin(i, sbackets_finish, assume_unique=True):
                    brac_square += i
                else:
                    key_square -= 1
                    sep.append("[" + brac_square + "]")
                    brac_square = ""
            else:
                if i.isdigit():
                    sep.append(i)
                elif i == "%":
                    sep.append("%" + s[count + 1] + s[count + 2])
                    count += 2
                elif i.islower():
                    try:
                        flag1 = False
                        if np.isin(i + s[count + 1], aromaticity, assume_unique=True):
                            flag1 = True
                    except Exception:
                        flag1 = False
                    try:
                        flag2 = False
                        if np.isin(i, aromaticity, assume_unique=True):
                            flag2 = True
                    except Exception:
                        flag2 = False
                    try:
                        flag3 = False
                        if elements(s[count - 1] + i, "Name"):
                            flag3 = True
                    except Exception:
                        flag3 = False

                    if flag1:
                        # print("aaa", i + s[count + 1])
                        sep.append(i + s[count + 1])
                        count += 1
                    elif flag2 and not flag3:
                        sep.append(i)
                    elif flag2 and flag3 and flag0:
                        sep.append(i)
                    elif flag3 and not flag0:
                        # sep.pop(-1)
                        sep.append(s[count - 1] + i)
                    elif flag3 and flag0 and not flag2:
                        sep.pop(-1)
                        sep.append(s[count - 1] + i)

                elif np.isin(i, sbackets_start, assume_unique=True):
                    key_square += 1
                else:
                    try:
                        flag0 = False
                        if elements(i, "Name"):
                            flag0 = True
                    except Exception:
                        flag0 = False
                    if flag0 or not i.isalpha():
                        sep.append(i)
            count += 1

        # bond packing
        if pack_bonds:
            i = 0
            while i < len(sep):
                if np.isin(sep[i], bonds, assume_unique=True):
                    sep[i] = sep[i] + sep[i + 1]
                    sep.pop(i + 1)
                i += 1
        return sep

    def smilesVOC(
        self, s: str, pack_bonds: bool = False, n_jobs: int = -1
    ) -> List[str]:
        """Function to create a "vocabulary" out of a list of SMILES.
        The SMILES are first segmented using the smilesSEP function.

        Parameters
        ----------
        s : list
            List containing all the SMILES.
        pack_bonds : bool
            if True, explicity bonds from the SMILES will be packed
            with the next element of the sequence. Ex.: C=C will result in
            [C, =C] instead of [C, =, C]
        n_jobs : integer
            The number of jobs to parallelize the calculation. If -1, all available
            cores will be used.

        Returns
        -------
        list
            List with the vocabulary (unique elements from the list of SMILES)

        """

        def compute(i):
            v = []
            t = self.smilesSEP(s[i], pack_bonds=pack_bonds)
            for ii in t:
                v.append(ii)
            return v[:]

        temp = Parallel(
            n_jobs=n_jobs,
            verbose=1,
            max_nbytes="200M",
            # batch_size=64,
            backend="threading",  # 1035904 / 10.4 min for threading
            # pre_dispatch=128,
        )(delayed(compute)(i) for i in range(len(s)))
        return list(np.unique([item for sublist in temp for item in sublist]))  # type: ignore

    def letterToIndex(self, entry: str, vocab: List[str]) -> int:
        """Return the letter/index based on vocab. 0 will be returned
        if the entry is not part of vocab

        Args:
            entry: a unique element from the SMILES
            vocab: the vocab list

        Returns:
            the corresponding index
        """
        try:
            n = vocab.index(entry) + 1
        except Exception:
            n = 0
        return n

    def smilesToSequence(
        self, entry: str, vocab: List[str], pack_bonds: bool = False
    ) -> List[List[int]]:
        """Translates the SMILES into a index list based on the vocab.

        Args:
            entry: the SMILES string

        Returns:
            sequence of indexed SMILES
        """
        return [
            [self.letterToIndex(i, vocab)]
            for i in self.smilesSEP(entry, pack_bonds=pack_bonds)
        ]

    def capacity_check(self, smiles: str) -> Tuple[int, float]:
        """Read a SMILES string and return the theoretical lithiation capacity
        in mAh/g based on a simple redox centers inference. Useful for Li-ion
        batteries.
        Uses the pysmiles package (https://github.com/pckroon/pysmiles) to
        get the number of implicit hydrogens.

        Parameters
        ----------
        smiles : string
            The SMILES string

        Returns
        -------
        float
            Theoretical capacity in mAh/g

        """
        redox_units = {"=O": 1, "=N": 1, "#N": 2, "S": 1, "n": 1}
        kk = redox_units.keys()

        redox_centers = 0
        for i in self.smilesSEP(smiles, pack_bonds=True):
            if i in kk:
                redox_centers += redox_units[i]

        unique, counts = np.unique(self.smilesSEP(smiles), return_counts=True)
        tt = dict(zip(unique, counts))
        molar_mass = 0
        for i in tt:
            if i.isalpha():
                n = tt[i]
                if len(i) > 1:
                    i = i[0].upper() + i[1].lower()
                else:
                    i = i.upper()
                molar_mass += elements(i, "A") * n
        li_mass = redox_centers * float(elements("Li", "A"))
        hydrogens = len(
            read_smiles(smiles, reinterpret_aromatic=False).nodes(data="hcount")  # type: ignore
        )
        molar_mass += li_mass + hydrogens * float(elements("H", "A"))
        Faraday = 96485.33212
        return redox_centers, (redox_centers * Faraday) / (3.6 * molar_mass)

    def vocab_symbols(self, smiles: str) -> List[str]:
        vocab_symbols = []
        vocab = self.smilesVOC(smiles, n_jobs=-1)
        for i in vocab:
            if "[" in i:
                if i[-2].islower():
                    vocab_symbols.append(i[1:-1])
                else:
                    vocab_symbols.append(i[1:-2].upper())
            elif i.isalpha():
                if i[0].islower():
                    vocab_symbols.append(i.upper())
                else:
                    vocab_symbols.append(i)
        return list(np.unique(vocab_symbols))
