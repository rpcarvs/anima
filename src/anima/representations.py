"""
Representations for molecular and periodic systems

Created by © Rodrigo darvalho 2019
Maintained by © Rodrigo Carvalho
"""

import gzip
import math
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.preprocessing import scale as sc


class Tools:
    def mol_xyz(self, mol_in, name_out):
        """Save xyz structure stored in a dataframe to xyz file
        Output in Cartesian (A)

        Parameters
        ----------
        mol_in : xyz in dataframe
            Obtained by reading a xyz file with the read_xyz function
        name_out : string
            Name/path of the output xyz

        """

        atoms = mol_in["atom"]
        Natoms = len(atoms)

        # saving
        f = open(name_out, "w")
        f.write(str(Natoms))
        f.write("\n COMMENTS\n")
        for i in range(0, len(atoms), 1):
            v = list(mol_in.iloc[i][1:])
            f.write(
                atoms[i]
                + "     "
                + str(v[0])
                + "     "
                + str(v[1])
                + "     "
                + str(v[2])
            )
            f.write("\n")
        f.close()

    def xyz_posc(self, mol_in, name_out, box_inc=15, selective_dynamics=False):
        """Save a molecule from mol_in to VASP/POSCAR format with a reasonable box
        Output in Cartesian

        Parameters
        ----------
        mol_in : xyz in dataframe
            Obtained by reading a xyz file with the read_xyz function
        name_out : string
            Name/path for the POSCAR
        box_inc : float
            Number in Angstrom to expand the cell. The molecule will be located
            at the center of the box.
        selective_dynamics : bool
            Check if the VASP selective dynamics related info should be written
        """

        dist = []
        for atom_j in range(0, len(mol_in["atom"]), 1):
            for atom_i in range(0, len(mol_in["atom"]), 1):
                dist.append(
                    math.sqrt(
                        (mol_in.iloc[atom_i]["x"] - mol_in.iloc[atom_j]["x"]) ** 2
                        + (mol_in.iloc[atom_i]["y"] - mol_in.iloc[atom_j]["y"]) ** 2
                        + (mol_in.iloc[atom_i]["z"] - mol_in.iloc[atom_j]["z"]) ** 2
                    )
                )
        box = max(dist) + box_inc
        atoms = mol_in["atom"].unique()
        f = open(name_out, "w")
        for name in atoms:
            f.write(str(name) + " ")
        f.write("\n")
        f.write("1.0 \n")
        f.write(str(box) + " 0.0 0.0 \n")
        f.write("0.0 " + str(box) + " 0.0 \n")
        f.write("0.0 0.0 " + str(box) + "\n")
        for name in atoms:
            f.write(str(name) + " ")
        f.write("\n")
        for name in atoms:
            f.write(str(len(mol_in[mol_in["atom"] == name])) + " ")
        f.write("\n")
        if selective_dynamics == True:
            f.write("Selective dynamics\n")
        f.write("Cartesian\n")
        xm = mol_in["x"].mean()
        ym = mol_in["y"].mean()
        zm = mol_in["z"].mean()
        for atom in range(0, len(mol_in["atom"]), 1):
            f.write(str(mol_in.iloc[atom]["x"] - xm + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["y"] - ym + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["z"] - zm + box / 2) + " ")
            if selective_dynamics == True:
                f.write(" T T T ")
            f.write("\n")

    def obsolete_xyz_posc_sel(self, mol_in, name_out):
        """Save xyz to POSCAR with a reasonable box
        Output in Cartesian with Selective Dynamics tag

        """
        dist = []
        for atom_j in range(0, len(mol_in["atom"]), 1):
            for atom_i in range(0, len(mol_in["atom"]), 1):
                dist.append(
                    math.sqrt(
                        (mol_in.iloc[atom_i]["x"] - mol_in.iloc[atom_j]["x"]) ** 2
                        + (mol_in.iloc[atom_i]["y"] - mol_in.iloc[atom_j]["y"]) ** 2
                        + (mol_in.iloc[atom_i]["z"] - mol_in.iloc[atom_j]["z"]) ** 2
                    )
                )
        box = max(dist) + 15.0
        atoms = mol_in["atom"].unique()
        f = open(name_out, "w")
        for name in atoms:
            f.write(str(name) + " ")
        f.write("\n")
        f.write("1.0 \n")
        f.write(str(box) + " 0.0 0.0 \n")
        f.write("0.0 " + str(box) + " 0.0 \n")
        f.write("0.0 0.0 " + str(box) + "\n")
        for name in atoms:
            f.write(str(name) + " ")
        f.write("\n")
        for name in atoms:
            f.write(str(len(mol_in[mol_in["atom"] == name])) + " ")
        f.write("\n")
        f.write("Selective dynamics\n")
        f.write("Cartesian\n")
        xm = mol_in["x"].mean()
        ym = mol_in["y"].mean()
        zm = mol_in["z"].mean()
        for atom in range(0, len(mol_in["atom"]), 1):
            f.write(str(mol_in.iloc[atom]["x"] - xm + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["y"] - ym + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["z"] - zm + box / 2) + " ")
            f.write(" T T T \n")

    def posc_xyz(self, posc_file, name_out):
        """Convert a VASP/POSCAR-format file into a xyz file.

        Parameters
        ----------
        posc_file : string
            Name/path of the POSCAR file
        name_out : type
            Name/path of the xyz file

        """
        file_posc = open(posc_file, mode="r")
        x = file_posc.read().splitlines()
        end = file_posc.tell()
        f = open(name_out, mode="w")
        lat = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
        coord = []
        coord_cart = []
        atoms = x[5].split()
        Nspecies = x[6].split()
        Natoms = [int(i) for i in Nspecies]
        Natoms = sum(Natoms)
        n = 3
        for n in (2, 3, 4):
            for i in (0, 1, 2):
                lat[n - 1][i] = x[n].split()[i]
            n = n + 1
        if list(x[7])[0] == "S":
            for n in range(9, 9 + Natoms, 1):
                coord.append(x[n].split())
            for i in range(0, Natoms, 1):
                coord[i] = list(filter(lambda a: (a != "T" and a != "F"), coord[i]))
            coord_mat = np.array(coord, dtype=float)
            lat_mat = np.array(list(lat.values()), dtype=float)
            for i in range(0, Natoms, 1):
                coord_cart.append(np.matmul(coord_mat[i], lat_mat))
            # SAVING
            f.write(str(Natoms))
            f.write("\n COMMENTS\n")
            k = 0
            for i in range(0, len(atoms), 1):
                for j in range(0, int(Nspecies[i])):
                    f.write(str(atoms[i]) + "     ")
                    f.write(str(coord_cart[k][0]) + "     ")
                    f.write(str(coord_cart[k][1]) + "     ")
                    f.write(str(coord_cart[k][2]) + "     ")
                    f.write("\n")
                    k += 1
        else:
            for n in range(8, 8 + Natoms, 1):
                coord.append(x[n].split())
            coord_mat = np.array(coord, dtype=float)
            lat_mat = np.array(list(lat.values()), dtype=float)
            for i in range(0, Natoms, 1):
                coord_cart.append(np.matmul(coord_mat[i], lat_mat))
            # SAVING
            f.write(str(Natoms))
            f.write("\n COMMENTS\n")
            k = 0
            for i in range(0, len(atoms), 1):
                for j in range(0, int(Nspecies[i])):
                    f.write(str(atoms[i]) + "     ")
                    f.write(str(coord_cart[k][0]) + "     ")
                    f.write(str(coord_cart[k][1]) + "     ")
                    f.write(str(coord_cart[k][2]) + "     ")
                    f.write("\n")
                    k += 1

    def element(self, symbol, function):
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

    def obsolete_get_en(self, file):
        """
        Get energies from file of type (step x energy) from MD in pd.DataFrame
        format step must be an integer like 1,2,3,4,5,6...
        The columns can have names.
        """
        file_en = open(file, mode="r")
        en = file_en.read().splitlines()
        file_en.close()
        if list(en[0])[0].isdigit() == True:
            energ = pd.read_csv(file, delim_whitespace=True, header=None)
        else:
            energ = pd.read_csv(file, delim_whitespace=True)
        return energ

    def read_xyz(self, file, sort=True, atomic_number=False):
        """Function to read and sort atoms in a xyz file.
        The structure will be stored as a pandas.DataFrame.

        Parameters
        ----------
        file : string
            Name/path of the xyz file
        sort : bool
            If True the atoms are sorted following the criteria order:
            Aphabetic order, x-coordinate, y-coordinate, z-coordinate.
        atomic_number : bool
            If True the atomic number will be also stored as an extra column.

        Returns
        -------
        pandas.DataFrame
            xyz-structure encoded in columns Atom, X, Y, Z, ATOMIC_NUMBER (optional)

        """

        mol_in = pd.read_csv(
            file,
            delim_whitespace=True,
            skiprows=2,
            header=None,
            names=("atom", "x", "y", "z"),
        )
        if sort == True:
            mol_in.sort_values(by=["atom", "x", "y", "z"], inplace=True)
            mol_in.reset_index(drop=True, inplace=True)
        if atomic_number == True:
            mol_in["Z"] = [
                self.element(mol_in["atom"][i], "Z") for i in range(len(mol_in))
            ]
        return mol_in

    def read_pdb(self, file, sort=True, atomic_number=False):
        """Function to read and sort atoms in a pdb file.
        The structure will be stored as a pandas.DataFrame.
        Only atom species and coordinates are read.

        Parameters
        ----------
        file : string
            Name/path of the pdb file
        sort : bool
            If True the atoms are sorted following the criteria order:
            Aphabetic order, x-coordinate, y-coordinate, z-coordinate.
        atomic_number : bool
            If True the atomic number will be also stored as an extra column.

        Returns
        -------
        pandas.DataFrame
            xyz-structure encoded in columns Atom, X, Y, Z, ATOMIC_NUMBER (optional)

        """

        mol_in = pd.read_csv(
            file,
            delim_whitespace=True,
            skiprows=1,
            header=None,
        )
        mol_in = pd.DataFrame(mol_in[mol_in[0] == "HETATM"].drop([0, 1, 3, 7], axis=1))
        mol_in.columns = ("atom", "x", "y", "z")

        if sort == True:
            mol_in.sort_values(by=["atom", "x", "y", "z"], inplace=True)
            mol_in.reset_index(drop=True, inplace=True)
        if atomic_number == True:
            mol_in["Z"] = [
                self.element(mol_in["atom"][i], "Z") for i in range(len(mol_in))
            ]
        return mol_in

    def pad_along_axis(self, array: np.ndarray, target_length, axis=0):
        """
        Zero-pad the input array along the desired dimension/axis.
        from: https://stackoverflow.com/questions/19349410/
        """

        pad_size = target_length - array.shape[axis]
        axis_nb = len(array.shape)

        if pad_size < 0:
            return a

        npad = [(0, 0) for x in range(axis_nb)]
        npad[axis] = (0, pad_size)

        b = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

        return b

    def calculator(
        self,
        calc_type,
        function,
        structures,
        n_jobs=-1,
        verbose=5,
        max_nbytes="200M",
        batch_size=320,
        backend="threading",
        pre_dispatch="640",
        final_pad=True,
        **kwargs,
    ):
        """Tool to compute the representations in parallel. calc_type can be CM or MBTR.

        Parameters
        ----------
        calc_type : string
            Specify the representation. (only CM or MBTR implemented so far)
        function : callable
            should be the representation.CM or representation.MBTR callable object
        structures : list
            list of structures to which the representations will be computed
        n_jobs : check the joblib.Parallel documentation.
        verbose : check the joblib.Parallel documentation.
        max_nbytes : check the joblib.Parallel documentation.
        batch_size : check the joblib.Parallel documentation.
        backend : check the joblib.Parallel documentation.
        pre_dispatch : check the joblib.Parallel documentation.
        final_pad : type
            set True if you want the final set of representations to be zero-padded
        **kwargs : list
            representation-specific arguments

        Returns
        -------
        list
            Description of returned object.

        """
        n_elem = len(structures)

        """
        Tool to compute the representations in parallel. calc_type can be CM or MBTR.
        calc_type: should be either CM or MBTR.
        function: should be the representation.CM or representation.MBTR callable object
        sctructures: list of structures to which the representations will be computed
        n_jobs, verbose, max_nbytes,batch_size, backend and pre_dispatch follows the syntax of joblib.Parallel module.
        final_pad: set True if you want the final set of representations to be zero-padded
        """

        # define the calculator function and compute
        if calc_type == "CM":

            def compute(i, structure=structures):
                temp = function(structure[i])
                return temp

            all_cm = Parallel(
                n_jobs=n_jobs,
                verbose=verbose,
                max_nbytes=max_nbytes,
                batch_size=batch_size,
                backend=backend,
                pre_dispatch=pre_dispatch,
            )(delayed(compute)(i) for i in list(structures.keys()))

        elif calc_type == "MBTR":

            def compute(i, structure=structures, **kwargs):
                temp = []
                xx, f, elem, f_pad = function(structure=structure[i], **kwargs)
                temp.append(xx)
                temp.append(f)
                temp.append(elem)
                temp.append(f_pad)
                return temp

            all_mbtr = Parallel(
                n_jobs=n_jobs,
                verbose=verbose,
                max_nbytes=max_nbytes,
                batch_size=batch_size,
                backend=backend,
                pre_dispatch=pre_dispatch,
            )(delayed(compute)(i, **kwargs) for i in list(structures.keys()))

            x = []
            f = []
            elem = []
            f_pad = []
            for i in range(len(all_mbtr)):
                x.append(all_mbtr[i][0])
                f.append(all_mbtr[i][1])
                elem.append(all_mbtr[i][2])
                f_pad.append(all_mbtr[i][3])

        # zero-pad all calculated structures to have the same dimension
        if final_pad == True:
            if calc_type == "CM":
                mx = []
                for i in range(n_elem):
                    mx.append(np.shape(all_cm[i])[1])
                mx = np.max(mx)

                all_cm_pad = []
                for i in range(n_elem):
                    temp = self.pad_along_axis(all_cm[i], mx, axis=1)
                    all_cm_pad.append(self.pad_along_axis(temp, mx, axis=0))

                all_cm_pad = np.array(all_cm_pad)

                return all_cm, all_cm_pad

            elif calc_type == "MBTR":
                mx = []
                for i in range(n_elem):
                    mx.append(np.shape(f_pad[i])[1])
                mx = np.max(mx)

                all_mbtr_pad = []
                for i in range(n_elem):
                    all_mbtr_pad.append(self.pad_along_axis(f_pad[i], mx, axis=1))
                all_mbtr_pad = np.array(all_mbtr_pad)

                return x, f, elem, f_pad, all_mbtr_pad
        else:
            if calc_type == "CM":
                return all_cm

            elif calc_type == "MBTR":
                return x, f, elem, f_pad

    def rep_save(self, kind, data, force_rewrite=False):
        """Function to save the calculated representation.

        Parameters
        ----------
        kind : string
            CM or MBTR
        data : list
            Data to be saved in list format. Ex:
            CM:
                data = [cm, cm_pad]
            MBTR:
                data = [x, f, elem, f_pad, all_mbtr_pad]
        force_rewrite : bool
            Delete old saved representations.

        Returns
        -------
        Nothing

        """
        kind = kind.upper()

        act_dir = os.getcwd()

        if kind == "CM":
            if force_rewrite == True:
                shutil.rmtree("cm_data")
                print("Warning! Old folders have been deleted!")
            else:
                if os.path.exists("cm_data") == True:
                    return print("Folder exist!!")
            os.mkdir("cm_data")
            os.chdir("cm_data")
        elif kind == "MBTR":
            if force_rewrite == True:
                shutil.rmtree("mbtr_data")
                print("Warning! Old folders have been deleted!")
            else:
                if os.path.exists("mbtr_data") == True:
                    return print("Folder exist!!")
            os.mkdir("mbtr_data")
            os.chdir("mbtr_data")
        else:
            return print("Error, define the representation kind!")

        max_bytes = 2**31 - 1

        for i in range(len(data)):
            bytes_out = pickle.dumps(data[i], protocol=pickle.HIGHEST_PROTOCOL)
            with open("data" + str(i), "wb") as filename:
                for idx in range(0, len(bytes_out), max_bytes):
                    filename.write(bytes_out[idx : idx + max_bytes])
            with open("data" + str(i), "rb") as filename:
                with gzip.open("data" + str(i) + ".gz", "wb") as f:
                    shutil.copyfileobj(filename, f)
            os.remove("data" + str(i))

        """for i in range(len(data)):
            bytes_out = pickle.dumps(data[i],protocol=4)
            with open("data" + str(i), "wb") as filename:
                pickle.dump(data[i], filename, protocol=4)
            with open("data" + str(i), "rb") as filename:
                with gzip.open("data" + str(i) + ".gz", "wb") as f:
                    shutil.copyfileobj(filename, f)
            os.remove("data" + str(i))"""

        os.chdir(act_dir)

    def rep_load(self, kind, dim=None, sdir=None):
        """Function to load the saved representation. Informe the kind of representation (CM or MBTR) and the dir where it is stored. If the dir is not informed, the code will assume the default name for the selected representation.

        Parameters
        ----------
        kind : string
            CM or MBTR
        dim : list of integers or None
            Select the data to be loaded based on the kind.
                CM: 0-non padded data
                    1-padded data
                MBTR: 0-array of x values for each dimension
                      1-the actual mbtr
                      2-list of elements (single, pairs, etc) that compose the mbtr
                      3-mbtr individually padded
                      4-mbtr globally padded
        sdir : string
            dir where the saved representation is stored.

        Returns
        -------
        List of loaded data.

        """
        kind = kind.upper()
        if sdir == None:
            if kind == "CM":
                sdir = "cm_data"
            elif kind == "MBTR":
                sdir = "mbtr_data"

        if os.path.exists(sdir) == False:
            return print("Folder does not exist!")

        act_dir = os.getcwd()
        os.chdir(sdir)
        l = len(os.listdir())

        data = []
        max_bytes = 2**31 - 1

        for i in range(l):
            if dim and i in dim:
                temp = b""
                input_size = os.path.getsize("data" + str(i) + ".gz")
                with gzip.open("data" + str(i) + ".gz", "rb") as filec:
                    while True:
                        reading = filec.read(max_bytes)
                        if not reading:
                            break
                        temp += reading
                # data2 = pickle.loads(temp)
                data.append(pickle.loads(temp))
                del temp

        """for i in range(l):
            bytes_in = bytearray(0)
            input_size = os.path.getsize("data" + str(i) + ".gz")
            with gzip.open("data" + str(i) + ".gz", "rb") as filec:
                for _ in range(0, input_size, max_bytes):
                    bytes_in += filec.read(max_bytes)
                data2 = pickle.loads(bytes_in)
                data.append(data2)"""

        """for i in range(l):
            with gzip.open("data" + str(i) + ".gz", "rb") as filec:
                data.append(pickle.load(filec))
            #with open("data" + str(i), "rb") as filename:
                #data.append(pickle.load(filename))
        """

        os.chdir(act_dir)

        return data


class Coulomb_Matrix:
    """
    Matthias Rupp, Alexandre Tkatchenko, Klaus-Robert Müller, O. Anatole von Lilienfeld:
    Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning, Physical Review Letters 108(5): 058301, 2012. DOI 10.1103/PhysRevLett.108.058301
    """

    def __init__(self):
        self.element = Tools().element

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
        elements = np.unique(atoms)
        Natoms = len(atoms)
        atm_lst = range(len(atoms))
        pos = []
        for i in range(Natoms):
            pos.append(np.array(mol_in.iloc[i][1:4], dtype="float"))
        pos = np.array(pos)

        # distances
        cmat = np.zeros((Natoms, Natoms))
        for j in range(Natoms):
            for i in range(j):
                zi = self.element(mol_in["atom"][i], "Z")
                zj = self.element(mol_in["atom"][j], "Z")
                cmat[i, j] = zi * zj / distance(pos[i], pos[j])

        # diagonal
        for i in range(Natoms):
            cmat[i, i] = 0.5 * (self.element(mol_in["atom"][i], "Z")) ** (2.4)

        if upper is not True:
            for j in range(Natoms):
                for i in range(j):
                    cmat[j, i] = cmat[i, j]
        return cmat

    def obsolete_cmatrix(self, mol_in):
        """
        Calculate the Coulomb Matrix of a sorted xyz molecule file from the funciton read_xyz.

        mol_in: structure read from a xyz file using the read_xyz function
        """
        # c_mat={}
        c_mat = []
        # c_mat=c_mat.fromkeys(range(0,len(mol_in['atom']),1))
        # for l in c_mat.keys():
        # for l in mol_in['atom']:
        #    mat_temp1=[]
        # mat[l]=[]
        for atom_j in range(0, len(mol_in["atom"]), 1):
            mat_temp = []
            for atom_i in range(0, len(mol_in["atom"]), 1):
                if atom_i < atom_j:
                    mat_temp.append(c_mat[atom_i][atom_j])
                else:
                    if atom_i == atom_j:
                        temp = 0.5 * (self.element(mol_in["atom"][atom_i], "Z")) ** (
                            2.4
                        )
                        # c_mat[atom_j].append(temp)
                        mat_temp.append(temp)
                    else:
                        dist = np.linalg.norm(
                            mol_in.iloc[atom_i][1:4] - mol_in.iloc[atom_j][1:4]
                        )
                        temp = (
                            (self.element(mol_in["atom"][atom_i], "Z"))
                            * (self.element(mol_in["atom"][atom_j], "Z"))
                            / dist
                        )
                        # c_mat[atom_j].append(temp)
                        mat_temp.append(temp)
            c_mat.append(mat_temp)
        c_mat = np.array(c_mat)
        return c_mat

    def obsolete_cmatrix2(self, mol_in):
        """
        Calculate the Coulomb Matrix of a sorted xyz molecule file from funciton read_mol.
        """
        # c_mat={}
        c_mat = []
        # c_mat=c_mat.fromkeys(range(0,len(mol_in['atom']),1))
        # for l in c_mat.keys():
        # for l in mol_in['atom']:
        #    mat_temp1=[]
        # mat[l]=[]
        for atom_j in range(0, len(mol_in["atom"]), 1):
            mat_temp = []
            for atom_i in range(0, len(mol_in["atom"]), 1):
                if atom_i < atom_j:
                    mat_temp.append(c_mat[atom_i][atom_j])
                else:
                    if atom_i == atom_j:
                        temp = 0.5 * (self.element(mol_in["atom"][atom_i], "Z")) ** (
                            2.4
                        )
                        # c_mat[atom_j].append(temp)
                        mat_temp.append(temp)
                    else:
                        dist = math.sqrt(
                            (mol_in.iloc[atom_i]["x"] - mol_in.iloc[atom_j]["x"]) ** 2
                            + (mol_in.iloc[atom_i]["y"] - mol_in.iloc[atom_j]["y"]) ** 2
                            + (mol_in.iloc[atom_i]["z"] - mol_in.iloc[atom_j]["z"]) ** 2
                        )
                        temp = (
                            (self.element(mol_in["atom"][atom_i], "Z"))
                            * (self.element(mol_in["atom"][atom_j], "Z"))
                            / dist
                        )
                        # c_mat[atom_j].append(temp)
                        mat_temp.append(temp)
            c_mat.append(mat_temp)
        c_mat = np.array(c_mat)
        return c_mat


class MBTR:
    """
    Class to compute the MBTR. Call the mbtr method.
    Huo, H., & Rupp, M. (2017). Unified representation for machine learning of molecules and crystals. arXiv preprint arXiv:1704.06439, 13754-13769.
    """

    def __init__(self):
        self.element = Tools().element

    def mbtr(
        self,
        k_idx,
        x_min,
        x_max,
        structure,
        acc=0.01,
        step=None,
        sigma=0.05,
        wt=np.repeat("quadratic", 5),
        scale=False,
        pad=False,
        cutoff=None,
        sparse=False,
    ):
        """
        Compute the mbtr
        > k_idx, x_min, x_max and wt must be in list [] format
        > k_idx is the k index of the mbtr. It is possible to choose like [1,3,4],[2,3],etc
        > x_min and x_max are lists of min and max value for the mbtr to each k index in the same order of k_idx list
        > acc is the accuracy to compute the mbtr. Caution with computational cost.
          if step is informed, acc is not used.
        > step is the number of points for each mbtr calculation. Must be a list with value for each k index.
        > sigma is the standart deviation for the normal distribution
        > wt is the weighting type function in list format for each k index. Options are unit, quadratic or exponential. If not supplied, quadratic will be used.
        > if scale is True, the output will be scaled to max unit.
        > structure is the structure which the mbtr will be computed. Use the Tool() class provided in this package to get structure in the correct format
        > if pad is True, a zero-padded copy of the mbtr will be supplied, useful for machine learning
        > cutoff is the cutoff in Angstroms which the mbtr calculation will stop. Valid for k=3,4
        > if sparse = True, sparce matrices will be used during 3-body and 4-body calculations to speed ups and free up memory. Use it to large structures.

        ------
        The output of this functions is either xx,f,elem or xx,f,elem,f_pad.

        Ex:
        x,f,elem = mbtr(k_idx,x_min,x_max,molecule,pad=False)

        Or:
        x,f,elem,f_pad = mbtr(k_idx,x_min,x_max,molecule,pad=True)

        x is the x value, useful to future plots.
        elem is the elements used in each iteration (one-body,two-boby,three-body,...) according to the k-value
        f is the mbtr, can be used together with xx to plots
        f_pad if the mbtr zero-padded, useful for machine learning

        """

        def distance(p1, p2):
            """
            Return the distance between two atoms locates at p1 and p2
            p1,p2 are atoms positions in cartesian coordinates
            """
            return np.linalg.norm(np.subtract(p2, p1))

        def gaussian(x, mean, sigma=0.1):
            """
            Return normal distribution
            """
            # f=lambda a: (1.0/np.sqrt(2*np.pi*sigma**2))*np.exp(-(a-mean)**2/(2*sigma**2))
            # return f(x)
            f = (1.0 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
                -(np.subtract(x.data, mean) ** 2) / (2 * sigma**2)
            )
            return f

        def g_k(k=None, p=None, d=None, X=None, atoms_list=None, cutoff=cutoff):
            """
            Compute the g_k in the MBTR
            """

            def angles(p1, p2, p3, d12, d13):
                """
                Compute angle p1p2p3
                p1,p2,p3 are atoms positions in cartesian coordinates
                """
                # if np.array_equal(p1,p2)==True or np.array_equal(p1,p3)==True or np.array_equal(p2,p3)==True:
                if (
                    (p1 == p2).all() == True
                    or (p1 == p3).all() == True
                    or (p2 == p3).all() == True
                ):
                    return 0.0
                # if arq(p1,p2)==True or arq(p1,p3)==True or arq(p2,p3)==True:
                #    return 0.0
                else:
                    # ba = p2-p1
                    # bc = p3-p1
                    b = p1
                    ba = np.subtract(p2, p1)
                    bc = np.subtract(p3, p1)
                    # dba= np.linalg.norm(ba)
                    # dbc= np.linalg.norm(bc)
                    # angle = np.arccos(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)))
                    cos_ang = np.dot(ba, bc) / (d12 * d13)
                    return cos_ang

            def atom_counts(X, atoms):
                """
                Return the number of specified X atom in the supplied atoms list
                X must be a string
                """
                return np.sum(atoms == X)

            def inv_distance(p1, p2):
                """
                Return the distance inverse between two atoms locates at p1 and p2
                p1,p2 are atoms positions in cartesian coordinates
                """
                return 1.0 / d if d != 0 else -4000

            def dihedral(p0, p1, p2, p3):
                """
                Return dihedral angle of points p0,p1,p2,p3
                p0,p1,p2,p3 are atoms positions in cartesian coordinates
                """
                # Praxeolitic formula
                # 1 sqrt, 1 cross product
                # FROM: https://stackoverflow.com/questions/20305272/
                if (
                    (p0 == p1).all() == True
                    or (p0 == p2).all() == True
                    or (p0 == p3).all() == True
                    or (p1 == p2).all() == True
                    or (p1 == p3).all() == True
                    or (p2 == p3).all() == True
                ):
                    return 0.0
                else:
                    # b0 = -1.0*(p1 - p0)
                    # b1 = p2 - p1
                    # b2 = p3 - p2
                    b0 = np.subtract(p0, p1)
                    b1 = np.subtract(p2, p1)
                    b2 = np.subtract(p3, p2)

                    # normalize b1 so that it does not influence magnitude of vector
                    # rejections that come next
                    b1 /= np.linalg.norm(b1)

                    # vector rejections
                    # v = projection of b0 onto plane perpendicular to b1
                    #   = b0 minus component that aligns with b1
                    # w = projection of b2 onto plane perpendicular to b1
                    #   = b2 minus component that aligns with b1
                    v = b0 - np.dot(b0, b1) * b1
                    w = b2 - np.dot(b2, b1) * b1

                    # angle between v and w in a plane is the torsion angle
                    # v and w may not be normalized but that's fine since tan is y/x
                    x = np.dot(v, w)
                    y = np.dot(np.cross(b1, v), w)
                    return np.cos(np.arctan2(y, x))

            if k == 0:
                return self.element(X, "Z")
            elif k == 1:
                if X is None or atoms_list is None:
                    return print("Error! No atoms list or atoms to count")
                else:
                    return atom_counts(X, atoms_list)
            elif k == 2:
                if d == 0.0:
                    return -4000
                else:
                    return 1.0 / d
            elif k == 3:
                return angles(p[0], p[1], p[2], d[0], d[1])
            elif k == 4:
                return dihedral(p[0], p[1], p[2], p[3])
            else:
                return print("Error! k must be 1, 2, 3 or 4 in g_k function")
                # exit()

        def corr(a, b, tp="kronecker"):
            """
            Return element correlation for a,b
            >> kronecker -> Kronecker Delta
            >> (NOT IMPLEMENTED) pearson -> Pearson product-moment correlation coefficients
            """
            if tp == "kronecker":
                return 1 if a == b else 0

        def weights(x, tp="unit"):
            """
            Return weigh value
            """
            if tp == "unit":
                return 1
            elif tp == "quadratic":
                return np.square(x)
            elif tp == "exponential":
                return np.exp(-x)

        def atoms_zz(k, atoms):
            """
            Return the zz atoms to iteration in mbtr equation
            """
            if k < 1 or k > 4:
                return print("Invalid k value")
            trial = []
            f = []
            var4 = [""] if k < 4 else atoms
            var3 = [""] if k < 3 else atoms
            var2 = [""] if k < 2 else atoms
            var1 = atoms
            for i in var4:
                for j in var3:
                    for k in var2:
                        for l in var1:
                            trial.append(i + " " + j + " " + k + " " + l)
            trial = list(np.core.defchararray.split(trial, " "))
            for ii in range(len(trial)):
                f.append(list(filter(None, trial[ii])))
            f = np.array(f, dtype="str")
            return np.unique(f, axis=0)

        def pad_along_axis(array: np.ndarray, target_length, axis=0):
            """
            https://stackoverflow.com/questions/19349410/
            """

            pad_size = target_length - array.shape[axis]
            axis_nb = len(array.shape)

            if pad_size < 0:
                return a

            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (0, pad_size)

            b = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

            return b

        # informations for fast indexing
        atoms = np.array(structure["atom"], dtype="str")
        elements = np.unique(atoms)
        Natoms = len(atoms)
        atm_lst = range(len(atoms))

        # get all positions for fast indexing
        # compute distances matrix
        if np.greater(k_idx, 1).any() == True:
            # positions array
            pos = []
            for i in range(Natoms):
                pos.append(np.array(structure.iloc[i][1:4], dtype="float"))
            pos = np.array(pos)

            # distances
            dd = np.zeros((Natoms, Natoms))
            for j in range(Natoms):
                for i in range(j):
                    dd[i, j] = distance(pos[i], pos[j])
                    dd[j, i] = dd[i, j]
            # for j in range(Natoms):
            #    for i in range(Natoms):
            #        if i < j:
            #            dd[i, j] = dd[j, i]
            #        else:
            #            dd[i, j] = distance(pos[i], pos[j])

        # starting calculation of MBTR
        f = []  # f is the mbtr
        xx = []  # xx is the x values, returned for future plots
        elements_it = []  # elements to iterate, returned for future plots
        idx = 0  # index related to the model supplied to .mbtr(...)

        for k in k_idx:
            if k == 0:
                """
                k=0 is a Z-distribution-like to generate a fingerprint of atomic values.
                It is not present in original MBTR proposed in [1]
                """
                f0 = []
                # define x_max,x_min
                steps = step[idx] if step else int((x_max[idx] - x_min[idx]) / acc)
                xx0 = np.linspace(x_min[idx], x_max[idx], steps)

                for x in xx0:
                    temp = 0.0
                    for atm in atoms:
                        temp += weights(x) * gaussian(x, g_k(k=0, X=atm), sigma=sigma)
                    f0.append(temp)
                f0 = np.array(f0)

                # scaling
                if scale == True:
                    f0 = sc(f0, axis=0, with_mean=False)

                f.append(np.reshape(f0, (1, np.shape(f0)[0])))
                xx.append(xx0)
                elements_it.append(None)

            if k == 1:
                f1 = []
                # define x_max,x_min
                steps = step[idx] if step else int((x_max[idx] - x_min[idx]) / acc)
                xx1 = np.linspace(x_min[idx], x_max[idx], steps)
                # define atoms to iterate
                atoms_iter = atoms_zz(k, atoms)

                # computing correlation and g_k arrays
                gg1 = []
                corr1 = []
                for zz in atoms_iter:
                    temp = []
                    temp2 = []
                    for atm in atoms:
                        temp.append(g_k(k=k, X=zz, atoms_list=atoms))
                        temp2.append(1 if corr(zz, atm) == 1 else 0)
                    gg1.append(temp)
                    corr1.append(temp2)
                gg1 = np.array(gg1)
                corr1 = np.array(corr1)

                # computing the mbtr
                for i in range(len(gg1)):
                    temp = []
                    for x in xx1:
                        temp.append(
                            np.sum(
                                weights(x, tp=wt[idx])
                                * gaussian(x, gg1[i], sigma=sigma)
                                * corr1[i]
                            )
                        )
                    f1.append(temp)
                f1 = np.array(f1)

                # scaling
                if scale == True:
                    f1 = sc(
                        f1.reshape(len(atoms_iter), len(xx1)), axis=1, with_mean=False
                    )
                f.append(f1)
                xx.append(xx1)
                elements_it.append(atoms_iter)

            if k == 2:
                f2 = []
                # define x_max,x_min
                steps = step[idx] if step else int((x_max[idx] - x_min[idx]) / acc)
                xx2 = np.linspace(x_min[idx], x_max[idx], steps)
                # atoms to iterate
                atoms_iter = atoms_zz(k, atoms)

                # computing correlation and g_k arrays
                gg2 = []
                corr2 = []
                for zz1, zz2 in atoms_iter:
                    temp = []
                    temp2 = []
                    for atm1 in atm_lst:
                        for atm2 in atm_lst:
                            if cutoff is not None:
                                if dd[atm2, atm1] > cutoff:
                                    temp.append(0.0)
                                    temp2.append(0.0)
                                else:
                                    temp.append(g_k(k=k, d=dd[atm2, atm1]))
                                    temp2.append(
                                        1
                                        if corr(zz1, atoms[atm1])
                                        * corr(zz2, atoms[atm2])
                                        == 1
                                        else 0
                                    )
                            else:
                                temp.append(g_k(k=k, d=dd[atm2, atm1]))
                                temp2.append(
                                    1
                                    if corr(zz1, atoms[atm1]) * corr(zz2, atoms[atm2])
                                    == 1
                                    else 0
                                )
                    gg2.append(temp)
                    corr2.append(temp2)
                gg2 = np.array(gg2)
                corr2 = np.array(corr2)

                # computing the mbtr
                for i in range(len(gg2)):
                    temp = []
                    for x in xx2:
                        temp.append(
                            np.sum(
                                weights(x, tp=wt[idx])
                                * gaussian(x, gg2[i], sigma=sigma)
                                * corr2[i]
                            )
                        )
                    f2.append(temp)
                f2 = np.array(f2)

                # scaling
                if scale == True:
                    f2 = sc(f2, axis=1, with_mean=False)
                f.append(f2)
                xx.append(xx2)
                elements_it.append(atoms_iter)

            # 3-body
            if k == 3:
                f3 = []
                # define x_max,x_min
                steps = step[idx] if step else int((x_max[idx] - x_min[idx]) / acc)
                xx3 = np.linspace(x_min[idx], x_max[idx], steps)
                # atoms to iterate
                atoms_iter = atoms_zz(k, atoms)

                if sparse == True:
                    # computing correlation and g_k arrays
                    ii = 0
                    n1 = []
                    n2 = []
                    temp = []
                    temp2 = []
                    for zz1, zz2, zz3 in atoms_iter:
                        ij = 0
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    if cutoff is not None:
                                        if (
                                            dd[atm3, atm1] > cutoff
                                            or dd[atm3, atm2] > cutoff
                                        ):
                                            pass
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[pos[atm3], pos[atm2], pos[atm1]],
                                                    d=[dd[atm3, atm2], dd[atm3, atm1]],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])
                                                * corr(zz2, atoms[atm2])
                                                * corr(zz3, atoms[atm3])
                                                == 1
                                                else 0
                                            )
                                            n2.append(ij)
                                            n1.append(ii)
                                            ij += 1
                                    else:
                                        temp.append(
                                            g_k(
                                                k=k,
                                                p=[pos[atm3], pos[atm2], pos[atm1]],
                                                d=[dd[atm3, atm2], dd[atm3, atm1]],
                                            )
                                        )
                                        temp2.append(
                                            1
                                            if corr(zz1, atoms[atm1])
                                            * corr(zz2, atoms[atm2])
                                            * corr(zz3, atoms[atm3])
                                            == 1
                                            else 0
                                        )
                                        n2.append(ij)
                                        n1.append(ii)
                                        ij += 1
                        ii += 1
                    gg3 = csr_matrix((temp, (n1, n2)))
                    corr3 = csr_matrix((temp2, (n1, n2)))

                    # computing the mbtr
                    for i in range(len(atoms_iter)):
                        temp = []
                        for x in xx3:
                            temp.append(
                                np.sum(
                                    weights(x, tp=wt[idx])
                                    * gaussian(x, gg3[i].A, sigma=sigma)
                                    * corr3[i].A
                                )
                            )
                        f3.append(temp)
                    f3 = np.array(f3)

                # no sparse
                else:
                    # computing correlation and g_k arrays
                    gg3 = []
                    corr3 = []
                    for zz1, zz2, zz3 in atoms_iter:
                        temp = []
                        temp2 = []
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    if cutoff is not None:
                                        if (
                                            dd[atm3, atm1] > cutoff
                                            or dd[atm3, atm2] > cutoff
                                        ):
                                            temp.append(0.0)
                                            temp2.append(0.0)
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[pos[atm3], pos[atm2], pos[atm1]],
                                                    d=[dd[atm3, atm2], dd[atm3, atm1]],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])
                                                * corr(zz2, atoms[atm2])
                                                * corr(zz3, atoms[atm3])
                                                == 1
                                                else 0
                                            )
                                    else:
                                        temp.append(
                                            g_k(
                                                k=k,
                                                p=[pos[atm3], pos[atm2], pos[atm1]],
                                                d=[dd[atm3, atm2], dd[atm3, atm1]],
                                            )
                                        )
                                        temp2.append(
                                            1
                                            if corr(zz1, atoms[atm1])
                                            * corr(zz2, atoms[atm2])
                                            * corr(zz3, atoms[atm3])
                                            == 1
                                            else 0
                                        )
                        gg3.append(temp)
                        corr3.append(temp2)
                    gg3 = np.array(gg3)
                    corr3 = np.array(corr3)

                    # computing the mbtr
                    for i in range(len(atoms_iter)):
                        temp = []
                        for x in xx3:
                            temp.append(
                                np.sum(
                                    weights(x, tp=wt[idx])
                                    * gaussian(x, gg3[i], sigma=sigma)
                                    * corr3[i]
                                )
                            )
                        f3.append(temp)
                    f3 = np.array(f3)

                # scaling
                if scale == True:
                    f3 = sc(f3, axis=1, with_mean=False)
                f.append(f3)
                xx.append(xx3)
                elements_it.append(atoms_iter)

            # 4-body
            if k == 4:
                f4 = []
                # define x_max,x_min
                steps = step[idx] if step else int((x_max[idx] - x_min[idx]) / acc)
                xx4 = np.linspace(x_min[idx], x_max[idx], steps)
                # atoms to iterate
                atoms_iter = atoms_zz(k, atoms)

                if sparse == True:
                    # computing correlation and g_k arrays
                    ii = 0
                    n1 = []
                    n2 = []
                    temp = []
                    temp2 = []
                    for zz1, zz2, zz3, zz4 in atoms_iter:
                        ij = 0
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    for atm4 in atm_lst:
                                        dis = np.array(
                                            [
                                                dd[atm1, atm2],
                                                dd[atm1, atm3],
                                                dd[atm1, atm4],
                                                dd[atm2, atm3],
                                                dd[atm2, atm4],
                                                dd[atm3, atm4],
                                            ]
                                        )
                                        if cutoff is not None:
                                            if np.any(dis > cutoff) == True:
                                                pass
                                            else:
                                                temp.append(
                                                    g_k(
                                                        k=k,
                                                        p=[
                                                            pos[atm1],
                                                            pos[atm2],
                                                            pos[atm3],
                                                            pos[atm4],
                                                        ],
                                                    )
                                                )
                                                temp2.append(
                                                    1
                                                    if corr(zz1, atoms[atm1])
                                                    * corr(zz2, atoms[atm2])
                                                    * corr(zz3, atoms[atm3])
                                                    * corr(zz4, atoms[atm4])
                                                    == 1
                                                    else 0
                                                )
                                                n2.append(ij)
                                                n1.append(ii)
                                                ij += 1
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[
                                                        pos[atm1],
                                                        pos[atm2],
                                                        pos[atm3],
                                                        pos[atm4],
                                                    ],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])
                                                * corr(zz2, atoms[atm2])
                                                * corr(zz3, atoms[atm3])
                                                * corr(zz4, atoms[atm4])
                                                == 1
                                                else 0
                                            )
                                            n2.append(ij)
                                            n1.append(ii)
                                            ij += 1
                        ii += 1

                    gg4 = csr_matrix((temp, (n1, n2)))
                    corr4 = csr_matrix((temp2, (n1, n2)))

                    # computing the mbtr
                    for i in range(len(atoms_iter)):
                        temp = []
                        for x in xx4:
                            temp.append(
                                np.sum(
                                    weights(x, tp=wt[idx])
                                    * gaussian(x, gg4[i].A, sigma=sigma)
                                    * corr4[i].A
                                )
                            )
                        f4.append(temp)
                    f4 = np.array(f4)

                # no sparse
                else:
                    # computing correlation and g_k arrays
                    gg4 = []
                    corr4 = []
                    for zz1, zz2, zz3, zz4 in atoms_iter:
                        temp = []
                        temp2 = []
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    for atm4 in atm_lst:
                                        dis = np.array(
                                            [
                                                dd[atm1, atm2],
                                                dd[atm1, atm3],
                                                dd[atm1, atm4],
                                                dd[atm2, atm3],
                                                dd[atm2, atm4],
                                                dd[atm3, atm4],
                                            ]
                                        )
                                        if cutoff is not None:
                                            if np.any(dis > cutoff) == True:
                                                temp.append(0.0)
                                                temp2.append(0.0)
                                            else:
                                                temp.append(
                                                    g_k(
                                                        k=k,
                                                        p=[
                                                            pos[atm1],
                                                            pos[atm2],
                                                            pos[atm3],
                                                            pos[atm4],
                                                        ],
                                                    )
                                                )
                                                temp2.append(
                                                    1
                                                    if corr(zz1, atoms[atm1])
                                                    * corr(zz2, atoms[atm2])
                                                    * corr(zz3, atoms[atm3])
                                                    * corr(zz4, atoms[atm4])
                                                    == 1
                                                    else 0
                                                )
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[
                                                        pos[atm1],
                                                        pos[atm2],
                                                        pos[atm3],
                                                        pos[atm4],
                                                    ],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])
                                                * corr(zz2, atoms[atm2])
                                                * corr(zz3, atoms[atm3])
                                                * corr(zz4, atoms[atm4])
                                                == 1
                                                else 0
                                            )

                        gg4.append(temp)
                        corr4.append(temp2)
                    gg4 = np.array(gg4)
                    corr4 = np.array(corr4)

                    # computing the mbtr
                    for i in range(len(gg4)):
                        temp = []
                        for x in xx4:
                            temp.append(
                                np.sum(
                                    weights(x, tp=wt[idx])
                                    * gaussian(x, gg4[i], sigma=sigma)
                                    * corr4[i]
                                )
                            )
                        f4.append(temp)
                    f4 = np.array(f4)

                # scaling
                if scale == True:
                    f4 = sc(f4, axis=1, with_mean=False)
                f.append(f4)
                xx.append(xx4)
                elements_it.append(atoms_iter)
            idx += 1

        # final mbtr
        f = np.array(f, dtype=object)
        f_pad = []
        if pad == True:
            temp = []
            for i in range(len(f)):
                temp.append(f[i].shape[0])
            dim1 = max(temp)
            temp = []
            for i in range(len(f)):
                temp.append(f[i].shape[1])
            dim2 = max(temp)
            temp = []
            for i in range(len(f)):
                temp.append(pad_along_axis(f[i], dim1, axis=0))
            for i in range(len(f)):
                f_pad.append(pad_along_axis(temp[i], dim2, axis=-1))
            f_pad = np.array(f_pad)

        xx = np.array(xx, dtype=object)

        return xx, f, elements_it, f_pad
