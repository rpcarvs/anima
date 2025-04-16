"""
IN DEVELOPMENT
Module to compute energy of systems based in pair interactions
Need the folder 'pot'.

Created by © Rodrigo Carvalho 2019
Mainteined by © Rodrigo Carvalho
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pymatgen as mg
from lib.representations import Coulomb_Matrix, Tools

# from keras.models import model_from_yaml
# from keras import backend

# backend.set_floatx("float64")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class PairPot:
    def __init__(self):
        prep = Coulomb_Matrix()
        print("pair-potentials module initialized")
        print(
            "Define the variables: \n   -pairs \n   -models \n   -norms \n   -atoms \nusing the appropriate functions before calling \nthe energy function"
        )

    # FUNCTIONS
    def distance(self, v1, v2):
        """
        Return the distance between two vectors
        """
        dist = math.sqrt(
            (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2
        )
        return dist

    # create atom pairs
    def create_pairs(self, mol_in, atoms, pot_dir):
        """
        Create the pairs array
        """
        pair = []
        inverse_pair = []
        final_pairs = []
        for j in range(len(atoms)):
            for i in range(j):
                at1 = mol_in.loc[i][0]
                at2 = mol_in.loc[j][0]
                pair.append(at1 + at2)
                inverse_pair.append(at2 + at1)
        index_p = np.unique(pair, return_index=True)[1]
        index_i = np.unique(inverse_pair, return_index=True)[1]
        pair = [pair[ind] for ind in sorted(index_p)]
        inverse_pair = [inverse_pair[ind] for ind in sorted(index_i)]
        for i in range(len(pair)):
            ph5 = os.path.isfile(pot_dir + "/" + pair[i] + ".h5")
            pym = os.path.isfile(pot_dir + "/" + pair[i] + ".yaml")
            pno = os.path.isfile(pot_dir + "/" + pair[i] + "Xnorm.dat")
            ih5 = os.path.isfile(pot_dir + "/" + inverse_pair[i] + ".h5")
            iym = os.path.isfile(pot_dir + "/" + inverse_pair[i] + ".yaml")
            ino = os.path.isfile(pot_dir + "/" + inverse_pair[i] + "Xnorm.dat")
            if ph5 == True and pym == True and pno == True:
                final_pairs.append(pair[i])
            elif ih5 == True and iym == True and ino == True:
                final_pairs.append(inverse_pair[i])
            else:
                print("Error! Impossible to find potential files")
                sys.exit()
        pairs = final_pairs
        return pairs

    # load models
    def load_models(self, pairs, pot_dir):
        """
        load potentials
        """
        models = {}
        for i in range(len(pairs)):
            h5 = pot_dir + "/" + pairs[i] + ".h5"
            yaml = pot_dir + "/" + pairs[i] + ".yaml"
            yaml_file = open(yaml)
            loaded_model_yaml = yaml_file.read()
            yaml_file.close()
            mod = None  # model_from_yaml(loaded_model_yaml)
            mod.load_weights(h5)
            mod.compile(loss="mae", optimizer="adam", metrics=["mse"])
            models[pairs[i]] = mod
            # load weights into new model
            print("Loaded model from disk")
        return models

    # load norms
    def load_norms(self, pairs, pot_dir):
        """
        load norm files
        """
        norms = {}
        for i in pairs:
            norm = pot_dir + "/" + i + "Xnorm.dat"
            file_pair = open(norm, mode="r")
            norms[i] = np.float64(file_pair.read().splitlines()[0])
            file_pair.close()
        return norms

    # compute system energy
    def energy(self, x, pairs=0, models=0, norms=0, atoms=0):
        """
        Compute the sum of pair energies
        """
        if pairs == 0 or models == 0 or norms == 0 or atoms == 0:
            print("Something went wrong!!")
            print("Please, check your variables and functions")
            sys.exit()
        loss = 0.0
        l = int(len(x) / 3)
        e1 = x[0:l].reshape(l)
        e2 = x[l : 2 * l].reshape(l)
        e3 = x[2 * l : 3 * l].reshape(l)
        molecule = pd.DataFrame({"atom": atoms, "x": e1, "y": e2, "z": e3})
        prep = Coulomb_Matrix()
        # prepare inputs
        pp = []
        ip = []
        # compute energy
        for jj in range(len(atoms)):
            for ii in range(jj):
                at1 = atoms[ii]
                at2 = atoms[jj]
                if at1 + at2 in pairs:
                    atm = [at1, at2]
                    v1 = molecule.iloc[ii][1:]
                    v2 = molecule.iloc[jj][1:]
                    xx = [v1[0], v2[0]]
                    yy = [v1[1], v2[1]]
                    zz = [v1[2], v2[2]]
                    mol = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                    matrix = np.array(prep.matrix(mol)) / norms[at1 + at2]
                    matrix = matrix.reshape(1, len(matrix) * len(matrix))
                    loss = loss + models[at1 + at2].predict(matrix)[0][0]
                else:
                    atm = [at2, at1]
                    v1 = molecule.iloc[jj][1:]
                    v2 = molecule.iloc[ii][1:]
                    xx = [v1[0], v2[0]]
                    yy = [v1[1], v2[1]]
                    zz = [v1[2], v2[2]]
                    mol = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                    matrix = np.array(prep.matrix(mol)) / norms[at2 + at1]
                    matrix = matrix.reshape(1, len(matrix) * len(matrix))
                    loss = loss + models[at2 + at1].predict(matrix)[0][0]

        return loss

    # compute system energy
    def forces(self, x, pairs=0, models=0, norms=0, atoms=0, dx=1e-10):
        """
        compute the total pair forces
        """
        if pairs == 0 or models == 0 or norms == 0 or atoms == 0:
            print("Something went wrong!!")
            print("Please, check your variables and functions")
            sys.exit()
        loss = 0.0
        l = int(len(x) / 3)
        e1 = x[0:l].reshape(l)
        e2 = x[l : 2 * l].reshape(l)
        e3 = x[2 * l : 3 * l].reshape(l)
        molecule = pd.DataFrame({"atom": atoms, "x": e1, "y": e2, "z": e3})
        prep = Coulomb_Matrix()
        # prepare inputs
        force = []
        # force=np.array(force,dtype='float64')
        # compute forces
        x_d = x
        for jj in range(len(atoms)):
            for ii in range(jj):
                at1 = atoms[ii]
                at2 = atoms[jj]
                if at1 + at2 in pairs:
                    atm = [at1, at2]
                    # f(x)
                    v1 = molecule.iloc[ii][1:]
                    v2 = molecule.iloc[jj][1:]
                    xx = [v1[0], v2[0]]
                    yy = [v1[1], v2[1]]
                    zz = [v1[2], v2[2]]
                    mol = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                    matrix = np.array(prep.matrix(mol)) / norms[at1 + at2]
                    matrix = matrix.reshape(1, len(matrix) * len(matrix))
                    f = models[at1 + at2].predict(matrix)[0][0]
                    # f(x+dx)
                    f_d = []
                    for ix in range(3):
                        v1d = np.array(v1)
                        v2d = np.array(v2)
                        v1d[ix] = v1d[ix] + dx
                        xx = [v1d[0], v2d[0]]
                        yy = [v1d[1], v2d[1]]
                        zz = [v1d[2], v2d[2]]
                        mol_d = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                        matrix = np.array(prep.matrix(mol_d)) / norms[at1 + at2]
                        matrix = matrix.reshape(1, len(matrix) * len(matrix))
                        f_d.append(models[at1 + at2].predict(matrix)[0][0])
                    f_d = np.array(f_d)
                    force.append(
                        np.append(
                            [at1 + at2, ii, jj], -(f_d - np.ones(len(f_d)) * f) / dx
                        )
                    )
                else:
                    atm = [at2, at1]
                    # f(x)
                    v1 = molecule.iloc[jj][1:]
                    v2 = molecule.iloc[ii][1:]
                    xx = [v1[0], v2[0]]
                    yy = [v1[1], v2[1]]
                    zz = [v1[2], v2[2]]
                    mol = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                    matrix = np.array(prep.matrix(mol)) / norms[at2 + at1]
                    matrix = matrix.reshape(1, len(matrix) * len(matrix))
                    f = models[at2 + at1].predict(matrix)[0][0]
                    # f(x+dx)
                    f_d = []
                    for ix in range(3):
                        v1d = np.array(v1)
                        v2d = np.array(v2)
                        v1d[ix] = v1d[ix] + dx
                        xx = [v1d[0], v2d[0]]
                        yy = [v1d[1], v2d[1]]
                        zz = [v1d[2], v2d[2]]
                        mol_d = pd.DataFrame({"atom": atm, "x": xx, "y": yy, "z": zz})
                        matrix = np.array(prep.matrix(mol_d)) / norms[at2 + at1]
                        matrix = matrix.reshape(1, len(matrix) * len(matrix))
                        f_d.append(models[at2 + at1].predict(matrix)[0][0])
                    f_d = np.array(f_d)
                    force.append(
                        np.append(
                            [at2 + at1, jj, ii], -(f_d - np.ones(len(f_d)) * f) / dx
                        )
                    )
        # force=[ np.float64(n) for n in force[i][1:4]]
        return force  # np.array(force,dtype='float64').reshape(int(len(force)/3),3)

    def sum_forces(self, atoms, force):
        """
        Return the total force on each atom
        """
        s = []
        for j in range(len(atoms)):
            temp = np.array([0, 0, 0], dtype="float64")
            for i in range(len(force)):
                # check
                # a -> +f
                # b -> -f
                trial = [int(n) for n in (force[i][1:3])]
                if j == trial[0]:
                    # print(j,'a',trial)
                    f = [np.float64(n) for n in force[i][3:]]
                    temp = np.add(temp, f, dtype="float64")
                elif j == trial[1]:
                    # print(j,'b',trial)
                    f = [-np.float64(n) for n in force[i][3:]]
                    temp = np.add(temp, f, dtype="float64")
            # print(j,temp)
            s.append(temp)
        return np.array(s, dtype="float64")

    def dynamics(
        self,
        x_start,
        atoms,
        forces,
        sum_forces,
        pairs,
        models,
        norms,
        steps=1000,
        unit=1,
        dt=0,
        save_step=False,
        save_best=True,
    ):
        """
        Dynamics module. The default number of steps is 1000 and the default time step is dt=0.1 fs
        You can define the time unit using unit=T where T can be 1 for seconds (default) or 1e-15 for
        fs or any other desired unit. In all cases, the default timestep is 0.01 fs. You can change this with
        tag dt. You have to define (or at least initialize) all the others flags for this module to work properly.
        Set save_step=True if you want to save all steps in history folder (default is False).
        The best structure (with the lowest sum of forces) is saved by default, set save_best=False to change this.
        """
        if (
            pairs == 0
            or models == 0
            or norms == 0
            or atoms == 0
            or forces == 0
            or sum_forces == 0
        ):
            print("Something went wrong!!")
            print("Please, check your variables and functions")
            sys.exit()
        prep = Tools()
        factor = 1.0364268824678108e-28 / (unit**2)
        if dt == 0:
            dt = 1e-16 / unit
        if save_step == True:
            if os.path.isdir("history") == True:
                os.system("rm -f history_back")
                os.system("mv history history_back")
                print(
                    "\n Found a history directory!\n It was renamed to history_back.\n Any existing folder named history_back has been deleted. Check your previous calculations!\n"
                )
            os.makedirs("history")
        # initial forces
        force = forces(x_start, pairs, models, norms, atoms)
        sum_f = sum_forces(atoms, force)
        total_forces_best = sum(
            [np.sqrt(np.dot(sum_f[o], sum_f[o])) for o in range(len(sum_f))]
        )
        # initialize variables
        x = [[] for i in range(len(atoms))]
        v = [[] for i in range(len(atoms))]
        [v[i].append(0) for i in range(len(atoms))]
        x_new = []
        mass = []
        x_ar = x_start
        lenn = int(len(x_start))
        # x0 and x1 of each atom
        for i in range(len(atoms)):
            mass.append(mg.Element(atoms[i]).data["Atomic mass"] * factor)
            pos = x_start.reshape(3, len(atoms)).T[
                i
            ]  # np.array(molecule_start.iloc[i][1:4],dtype="float64")
            # define x0
            x[i].append(pos)
            # compute x1
            temp = pos + v[i][0] * dt + sum_f[i] * dt**2 / (2 * mass[i])
            x[i].append(np.array(temp, dtype="float64"))
            x_new.append(x[i][-1])
        x_ar = np.reshape(
            [[x_new[im][jm] for im in range(len(atoms))] for jm in range(3)], (lenn, 1)
        )

        # with v0,x0,x1
        for it in range(2, steps):
            force_a = force
            sum_f_a = sum_f
            force = forces(x_ar, pairs, models, norms, atoms)
            sum_f = sum_forces(atoms, force)
            total_forces = sum(
                [np.sqrt(np.dot(sum_f[o], sum_f[o])) for o in range(len(sum_f))]
            )
            # print(it,sum_f.sum(axis=0))
            # print(it,total_forces)
            if total_forces < total_forces_best:
                total_forces_best = total_forces
                x_best = x_ar
            x_new = []
            for i in range(len(atoms)):
                # compute new v
                temp = v[i][it - 2] + (sum_f[i] + sum_f_a[i]) * dt / (2 * mass[i])
                v[i].append(np.array(temp, dtype="float64"))
                # compute new x
                temp = x[i][it - 1] + v[i][it - 1] * dt + sum_f[i] * dt**2 / (mass[i])
                x[i].append(np.array(temp, dtype="float64"))
                x_new.append(x[i][it])
            # new x_arr
            # print(x_new)
            x_ar = np.reshape(
                [[x_new[im][jm] for im in range(len(atoms))] for jm in range(3)],
                (lenn, 1),
            )
            # SAVE starting molecule for basin hopping
            if save_step == True:
                l = int(len(x_ar) / 3)
                e1 = x_ar[0:l].reshape(l)
                e2 = x_ar[l : 2 * l].reshape(l)
                e3 = x_ar[2 * l : 3 * l].reshape(l)
                mol = pd.DataFrame({"atom": atoms, "x": e1, "y": e2, "z": e3})
                prep.xyz_xyz(mol, "history/step_" + str(it) + ".xyz")
        # save best structure
        if save_best == True:
            l = int(len(x_best) / 3)
            e1 = x_best[0:l].reshape(l)
            e2 = x_best[l : 2 * l].reshape(l)
            e3 = x_best[2 * l : 3 * l].reshape(l)
            molecule_best = pd.DataFrame({"atom": atoms, "x": e1, "y": e2, "z": e3})
            prep.xyz_xyz(molecule_best, "Best.xyz")
