import gzip
import math
import os
import pickle
import shutil

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..utils import elements


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
        if selective_dynamics:
            f.write("Selective dynamics\n")
        f.write("Cartesian\n")
        xm = mol_in["x"].mean()
        ym = mol_in["y"].mean()
        zm = mol_in["z"].mean()
        for atom in range(0, len(mol_in["atom"]), 1):
            f.write(str(mol_in.iloc[atom]["x"] - xm + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["y"] - ym + box / 2) + " ")
            f.write(str(mol_in.iloc[atom]["z"] - zm + box / 2) + " ")
            if selective_dynamics:
                f.write(" T T T ")
            f.write("\n")

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
        _ = file_posc.tell()
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
                lat[n - 1][i] = x[n].split()[i]  # type: ignore
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
        if sort:
            mol_in.sort_values(by=["atom", "x", "y", "z"], inplace=True)
            mol_in.reset_index(drop=True, inplace=True)
        if atomic_number:
            mol_in["Z"] = [elements(mol_in["atom"][i], "Z") for i in range(len(mol_in))]
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

        if sort:
            mol_in.sort_values(by=["atom", "x", "y", "z"], inplace=True)
            mol_in.reset_index(drop=True, inplace=True)
        if atomic_number:
            mol_in["Z"] = [elements(mol_in["atom"][i], "Z") for i in range(len(mol_in))]
        return mol_in

    def pad_along_axis(self, array: np.ndarray, target_length, axis=0):
        """
        Zero-pad the input array along the desired dimension/axis.
        from: https://stackoverflow.com/questions/19349410/
        """

        pad_size = target_length - array.shape[axis]
        axis_nb = len(array.shape)

        if pad_size < 0:
            return array

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

            def compute(i, structure=structures):  # type: ignore
                temp = function(structure[i])
                return temp

            all_cm = Parallel(
                n_jobs=n_jobs,
                verbose=verbose,
                max_nbytes=max_nbytes,
                batch_size=batch_size,  # type: ignore
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
                batch_size=batch_size,  # type: ignore
                backend=backend,
                pre_dispatch=pre_dispatch,
            )(delayed(compute)(i, **kwargs) for i in list(structures.keys()))

            x = []
            f = []
            elem = []
            f_pad = []
            for i in range(len(all_mbtr)):  # type: ignore
                x.append(all_mbtr[i][0])  # type: ignore
                f.append(all_mbtr[i][1])  # type: ignore
                elem.append(all_mbtr[i][2])  # type: ignore
                f_pad.append(all_mbtr[i][3])  # type: ignore

        # zero-pad all calculated structures to have the same dimension
        if final_pad:
            if calc_type == "CM":
                mx = []
                for i in range(n_elem):
                    mx.append(np.shape(all_cm[i])[1])  # type: ignore
                mx = np.max(mx)

                all_cm_pad = []
                for i in range(n_elem):
                    temp = self.pad_along_axis(all_cm[i], mx, axis=1)  # type: ignore
                    all_cm_pad.append(self.pad_along_axis(temp, mx, axis=0))

                all_cm_pad = np.array(all_cm_pad)

                return all_cm, all_cm_pad  # type: ignore

            elif calc_type == "MBTR":
                mx = []
                for i in range(n_elem):
                    mx.append(np.shape(f_pad[i])[1])  # type: ignore
                mx = np.max(mx)

                all_mbtr_pad = []
                for i in range(n_elem):
                    all_mbtr_pad.append(self.pad_along_axis(f_pad[i], mx, axis=1))  # type: ignore
                all_mbtr_pad = np.array(all_mbtr_pad)

                return x, f, elem, f_pad, all_mbtr_pad  # type: ignore
        else:
            if calc_type == "CM":
                return all_cm  # type: ignore

            elif calc_type == "MBTR":
                return x, f, elem, f_pad  # type: ignore

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
            if force_rewrite:
                shutil.rmtree("cm_data")
                print("Warning! Old folders have been deleted!")
            else:
                if os.path.exists("cm_data"):
                    return print("Folder exist!!")
            os.mkdir("cm_data")
            os.chdir("cm_data")
        elif kind == "MBTR":
            if force_rewrite:
                shutil.rmtree("mbtr_data")
                print("Warning! Old folders have been deleted!")
            else:
                if os.path.exists("mbtr_data"):
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
        if not sdir:
            if kind == "CM":
                sdir = "cm_data"
            elif kind == "MBTR":
                sdir = "mbtr_data"

        if not os.path.exists(sdir):  # type: ignore
            return print("Folder does not exist!")

        act_dir = os.getcwd()
        os.chdir(sdir)  # type: ignore
        ll = len(os.listdir())

        data = []
        max_bytes = 2**31 - 1

        for i in range(ll):
            if dim and i in dim:
                temp = b""
                with gzip.open("data" + str(i) + ".gz", "rb") as filec:
                    while True:
                        reading = filec.read(max_bytes)
                        if not reading:
                            break
                        temp += reading
                # data2 = pickle.loads(temp)
                data.append(pickle.loads(temp))
                del temp

        os.chdir(act_dir)

        return data
