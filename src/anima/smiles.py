"""
A collection of tools to manipulate SMILES strings

Created by © Rodrigo Carvalho 2020
Maintained by © Rodrigo Carvalho
"""

import os

import networkx as nx
import numpy as np

## for openbabel < 3
import openbabel.pybel as pybel
import pysmiles
import torch
from joblib import Parallel, delayed
from pysmiles import read_smiles
from rdkit import Chem
from torch_geometric.data import Data

from .lib import xyz2mol

# for openbabel 2.4
# import pybel


class SMILES:
    def elements(self, symbol, function):
        """A simple function to return information of atomic elements.

        Parameters
        ----------
        symbol : string
            The element atomic symbol.
        function : string
            A for atomic mass, Z for atomic number and Name for its name.

        Returns
        -------
        Atomic number (int) for Z and name (string) for Name.

        """

        table = {
            "A": {
                "H": 1.00794,
                "He": 4.002602,
                "Li": 6.941,
                "Be": 9.012182,
                "B": 10.811,
                "C": 12.0107,
                "N": 14.0067,
                "O": 15.9994,
                "F": 18.9984032,
                "Ne": 20.1797,
                "Na": 22.98976928,
                "Mg": 24.305,
                "Al": 26.9815386,
                "Si": 28.0855,
                "P": 30.973762,
                "S": 32.065,
                "Cl": 35.453,
                "Ar": 39.948,
                "K": 39.0983,
                "Ca": 40.078,
                "Sc": 44.955912,
                "Ti": 47.867,
                "V": 50.9415,
                "Cr": 51.9961,
                "Mn": 54.938045,
                "Fe": 55.845,
                "Co": 58.933195,
                "Ni": 58.6934,
                "Cu": 63.546,
                "Zn": 65.409,
                "Ga": 69.723,
                "Ge": 72.64,
                "As": 74.9216,
                "Se": 78.96,
                "Br": 79.904,
                "Kr": 83.798,
                "Rb": 85.4678,
                "Sr": 87.62,
                "Y": 88.90585,
                "Zr": 91.224,
                "Nb": 92.90638,
                "Mo": 95.94,
                "Tc": 98.9063,
                "Ru": 101.07,
                "Rh": 102.9055,
                "Pd": 106.42,
                "Ag": 107.8682,
                "Cd": 112.411,
                "In": 114.818,
                "Sn": 118.71,
                "Sb": 121.760,
                "Te": 127.6,
                "I": 126.90447,
                "Xe": 131.293,
                "Cs": 132.9054519,
                "Ba": 137.327,
                "La": 138.90547,
                "Ce": 140.116,
                "Pr": 140.90465,
                "Nd": 144.242,
                "Pm": 146.9151,
                "Sm": 150.36,
                "Eu": 151.964,
                "Gd": 157.25,
                "Tb": 158.92535,
                "Dy": 162.5,
                "Ho": 164.93032,
                "Er": 167.259,
                "Tm": 168.93421,
                "Yb": 173.04,
                "Lu": 174.967,
                "Hf": 178.49,
                "Ta": 180.9479,
                "W": 183.84,
                "Re": 186.207,
                "Os": 190.23,
                "Ir": 192.217,
                "Pt": 195.084,
                "Au": 196.966569,
                "Hg": 200.59,
                "Tl": 204.3833,
                "Pb": 207.2,
                "Bi": 208.9804,
                "Po": 208.9824,
                "At": 209.9871,
                "Rn": 222.0176,
                "Fr": 223.0197,
                "Ra": 226.0254,
                "Ac": 227.0278,
                "Th": 232.03806,
                "Pa": 231.03588,
                "U": 238.02891,
                "Np": 237.0482,
                "Pu": 244.0642,
                "Am": 243.0614,
                "Cm": 247.0703,
                "Bk": 247.0703,
                "Cf": 251.0796,
                "Es": 252.0829,
                "Fm": 257.0951,
                "Md": 258.0951,
                "No": 259.1009,
                "Lr": 262,
                "Rf": 267,
                "Db": 268,
                "Sg": 271,
                "Bh": 270,
                "Hs": 269,
                "Mt": 278,
                "Ds": 281,
                "Rg": 281,
                "Cn": 285,
                "Nh": 284,
                "Fl": 289,
                "Mc": 289,
                "Lv": 292,
                "Ts": 294,
                "Og": 294,
            },
            "Z": {
                "H": 1,
                "He": 2,
                "Li": 3,
                "Be": 4,
                "B": 5,
                "C": 6,
                "N": 7,
                "O": 8,
                "F": 9,
                "Ne": 10,
                "Na": 11,
                "Mg": 12,
                "Al": 13,
                "Si": 14,
                "P": 15,
                "S": 16,
                "Cl": 17,
                "Ar": 18,
                "K": 19,
                "Ca": 20,
                "Sc": 21,
                "Ti": 22,
                "V": 23,
                "Cr": 24,
                "Mn": 25,
                "Fe": 26,
                "Co": 27,
                "Ni": 28,
                "Cu": 29,
                "Zn": 30,
                "Ga": 31,
                "Ge": 32,
                "As": 33,
                "Se": 34,
                "Br": 35,
                "Kr": 36,
                "Rb": 37,
                "Sr": 38,
                "Y": 39,
                "Zr": 40,
                "Nb": 41,
                "Mo": 42,
                "Tc": 43,
                "Ru": 44,
                "Rh": 45,
                "Pd": 46,
                "Ag": 47,
                "Cd": 48,
                "In": 49,
                "Sn": 50,
                "Sb": 51,
                "Te": 52,
                "I": 53,
                "Xe": 54,
                "Cs": 55,
                "Ba": 56,
                "La": 57,
                "Ce": 58,
                "Pr": 59,
                "Nd": 60,
                "Pm": 61,
                "Sm": 62,
                "Eu": 63,
                "Gd": 64,
                "Tb": 65,
                "Dy": 66,
                "Ho": 67,
                "Er": 68,
                "Tm": 69,
                "Yb": 70,
                "Lu": 71,
                "Hf": 72,
                "Ta": 73,
                "W": 74,
                "Re": 75,
                "Os": 76,
                "Ir": 77,
                "Pt": 78,
                "Au": 79,
                "Hg": 80,
                "Tl": 81,
                "Pb": 82,
                "Bi": 83,
                "Po": 84,
                "At": 85,
                "Rn": 86,
                "Fr": 87,
                "Ra": 88,
                "Ac": 89,
                "Th": 90,
                "Pa": 91,
                "U": 92,
                "Np": 93,
                "Pu": 94,
                "Am": 95,
                "Cm": 96,
                "Bk": 97,
                "Cf": 98,
                "Es": 99,
                "Fm": 100,
                "Md": 101,
                "No": 102,
                "Lr": 103,
                "Rf": 104,
                "Db": 105,
                "Sg": 106,
                "Bh": 107,
                "Hs": 108,
                "Mt": 109,
                "Ds": 110,
                "Rg": 111,
                "Cn": 112,
                "Nh": 113,
                "Fl": 114,
                "Mc": 115,
                "Lv": 116,
                "Ts": 117,
                "Og": 118,
            },
            "Name": {
                "H": "Hydrogen",
                "He": "Helium",
                "Li": "Lithium",
                "Be": "Beryllium",
                "B": "Boron",
                "C": "Carbon",
                "N": "Nitrogen",
                "O": "Oxygen",
                "F": "Fluorine",
                "Ne": "Neon",
                "Na": "Sodium",
                "Mg": "Magnesium",
                "Al": "Aluminum",
                "Si": "Silicon",
                "P": "Phosphorus",
                "S": "Sulfur",
                "Cl": "Chlorine",
                "Ar": "Argon",
                "K": "Potassium",
                "Ca": "Calcium",
                "Sc": "Scandium",
                "Ti": "Titanium",
                "V": "Vanadium",
                "Cr": "Chromium",
                "Mn": "Manganese",
                "Fe": "Iron",
                "Co": "Cobalt",
                "Ni": "Nickel",
                "Cu": "Copper",
                "Zn": "Zinc",
                "Ga": "Gallium",
                "Ge": "Germanium",
                "As": "Arsenic",
                "Se": "Selenium",
                "Br": "Bromine",
                "Kr": "Krypton",
                "Rb": "Rubidium",
                "Sr": "Strontium",
                "Y": "Yttrium",
                "Zr": "Zirconium",
                "Nb": "Niobium",
                "Mo": "Molybdenum",
                "Tc": "Technetium",
                "Ru": "Ruthenium",
                "Rh": "Rhodium",
                "Pd": "Palladium",
                "Ag": "Silver",
                "Cd": "Cadmium",
                "In": "Indium",
                "Sn": "Tin",
                "Sb": "Antimony",
                "Te": "Tellurium",
                "I": "Iodine",
                "Xe": "Xenon",
                "Cs": "Cesium",
                "Ba": "Barium",
                "La": "Lanthanum",
                "Ce": "Cerium",
                "Pr": "Praseodymium",
                "Nd": "Neodymium",
                "Pm": "Promethium",
                "Sm": "Samarium",
                "Eu": "Europium",
                "Gd": "Gadolinium",
                "Tb": "Terbium",
                "Dy": "Dysprosium",
                "Ho": "Holmium",
                "Er": "Erbium",
                "Tm": "Thulium",
                "Yb": "Ytterbium",
                "Lu": "Lutetium",
                "Hf": "Hafnium",
                "Ta": "Tantalum",
                "W": "Tungsten",
                "Re": "Rhenium",
                "Os": "Osmium",
                "Ir": "Iridium",
                "Pt": "Platinum",
                "Au": "Gold",
                "Hg": "Mercury",
                "Tl": "Thallium",
                "Pb": "Lead",
                "Bi": "Bismuth",
                "Po": "Polonium",
                "At": "Astatine",
                "Rn": "Radon",
                "Fr": "Francium",
                "Ra": "Radium",
                "Ac": "Actinium",
                "Th": "Thorium",
                "Pa": "Protactinium",
                "U": "Uranium",
                "Np": "Neptunium",
                "Pu": "Plutonium",
                "Am": "Americium",
                "Cm": "Curium",
                "Bk": "Berkelium",
                "Cf": "Californium",
                "Es": "Einsteinium",
                "Fm": "Fermium",
                "Md": "Mendelevium",
                "No": "Nobelium",
                "Lr": "Lawrencium",
                "Rf": "Rutherfordium",
                "Db": "Dubnium",
                "Sg": "Seaborgium",
                "Bh": "Bohrium",
                "Hs": "Hassium",
                "Mt": "Meitnerium",
                "Ds": "Darmstadtium",
                "Rg": "Roentgenium",
                "Cn": "Copernicium",
                "Nh": "Nihonium",
                "Fl": "Flerovium",
                "Mc": "Moscovium",
                "Lv": "Livermorium",
                "Ts": "Tennessine",
                "Og": "Oganesson",
            },
        }
        return table[function][symbol]

    def smiles_cleaner(self, s):
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

        return smi.split()[0].strip()

    def OB_standard_smiles(self, s, kekule=False):
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

        return smi.split()[0]

    def PS_fix(self, s):
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

    def deprecated_xyz_to_smiles(self, fname):
        """Read a xyz file and convert it into a SMILES stringself.
        ## requires xyz2mol.py (https://github.com/jensengroup/xyz2mol)

        Parameters
        ----------
        fname : type
            file name/path

        Returns
        -------
        string
            SMILES

        """
        return os.popen("xyz2mol.py " + str(fname)).read().splitlines()[0]

    def xyz_to_smiles(self, fname, chiral=False, charged_fragments=False):
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

    def standard_smiles(self, s, kekule=False, can=True):
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

    def get_hydrogens(self, fname):
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

    def smilesSEP(self, s, pack_bonds=False):
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
                        if self.elements(s[count - 1] + i, "Name"):
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
                        if self.elements(i, "Name"):
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

    def smilesVOC(self, s, pack_bonds=False, n_jobs=-1):
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
        return list(np.unique([item for sublist in temp for item in sublist]))

    def letterToIndex(self, entry, vocab):
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

    def smilesToSequence(self, entry, vocab, pack_bonds=False):
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

    def capacity_check(self, smiles):
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
                molar_mass += self.elements(i, "A") * n
        li_mass = redox_centers * self.elements("Li", "A")
        hydrogens = len(
            read_smiles(smiles, reinterpret_aromatic=False).nodes(data="hcount")
        )
        molar_mass += li_mass + hydrogens * self.elements("H", "A")
        Faraday = 96485.33212
        return redox_centers, (redox_centers * Faraday) / (3.6 * molar_mass)

    def vocab_symbols(self, smiles):
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

    def Smiles_To_Graph(self, s, vocab):
        """Read a SMILES string and convert it into a Graph. The Graph attributes
        will be based on a natural language processing approach, so a vocabullary
        needs to be supplied. The vocabullary for bonds is fixed as:
         ['AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE'].

        Parameters
        ----------
        s : string
            The SMILES string
        vocab : list
                vocabullary

        Returns
        -------
        networkx Graph

        """

        vocab_bonds = ["AROMATIC", "DOUBLE", "SINGLE", "TRIPLE"]
        G = nx.Graph()
        molecule = Chem.MolFromSmiles(s, sanitize=False)

        for atom in molecule.GetAtoms():
            temp = self.letterToIndex(atom.GetSymbol(), vocab)
            G.add_node(atom.GetIdx(), symbol=temp)
        for bond in molecule.GetBonds():
            temp = self.letterToIndex(bond.GetBondType().name, vocab_bonds)
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=temp)
        return G

    def batch_Smiles_To_Graph(self, smiles, vocab):
        """Function to parallelize the Smiles_To_Graph function.

        Parameters
        ----------
        smiles : list
            The list of SMILES string
        vocab : list
                vocabullary

        Returns
        -------
        list of networkx Graphs

        """

        def compute(i, vocab=vocab):
            try:
                return self.Smiles_To_Graph(i, vocab)
            except Exception:
                return None

        return Parallel(n_jobs=-1, verbose=1, max_nbytes="200M", backend="threading")(
            delayed(compute)(i) for i in smiles
        )

    def smiles_to_torch(self, vocab, smiles, targets=None):
        # sourcery skip: list-comprehension, use-assigned-variable
        """Return a list of torch_geometric Data format packing the graphs and
        targets (if supplied). Final graph attributes will be based on a natural
        language processing basing, thus the vocabullary. For bonds, the vobac is
        fixed as ['AROMATIC', 'DOUBLE', 'SINGLE', 'TRIPLE'].

        Requires PyTorch and Torch_geometric libraries.

        Parameters
        ----------
        vocab : list
                vocabullary
        smiles : list
                 The list of SMILES string
        targets : list
                  list of target values

        Returns
        -------
        list of torch_geonetric Data graphs and torch tensors.

        """

        # vocab = ['Br', 'C', 'Cl', 'F', 'Li', 'N', 'O', 'S']
        x = np.array(smiles)
        x_graphs = self.batch_Smiles_To_Graph(x, vocab)
        if targets is not None:
            y = targets
            # in case targets are a classification (yes/no based only!)
            if np.any(np.isin(["yes", "no"], list(targets))):
                y = torch.tensor(np.array(y.map(dict(yes=1, no=0))))
            y = torch.tensor(np.array(y))

        dataset = []
        for ii in range(len(x_graphs)):
            gg = x_graphs[ii]
            node_feats = []
            edge_feats = []
            for i in gg.nodes:
                node_feats.append(gg.nodes[i]["symbol"])
            for i in gg.edges:
                edge_feats.append(gg.get_edge_data(i[0], i[1])["bond_type"])
            edge_index = torch.tensor(np.array(gg.edges).T)
            node_feats = torch.tensor(node_feats)
            edge_feats = torch.tensor(edge_feats)
            if targets is not None:
                dataset.append(
                    Data(
                        x=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=y[ii],
                    )
                )
            else:
                dataset.append(
                    Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats)
                )

        return dataset
