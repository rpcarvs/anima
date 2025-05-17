import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import scale as sc

from ..utils import elements as elements_func


class MBTR:
    """
    Class to compute the MBTR. Call the mbtr method.
    Huo, H., & Rupp, M. (2017). Unified representation for machine learning of molecules and crystals. arXiv preprint arXiv:1704.06439, 13754-13769.
    """

    def __init__(self):
        pass

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
                if (p1 == p2).all() or (p1 == p3).all() or (p2 == p3).all():
                    return 0.0
                # if arq(p1,p2)==True or arq(p1,p3)==True or arq(p2,p3)==True:
                #    return 0.0
                else:
                    # ba = p2-p1
                    # bc = p3-p1
                    # b = p1
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
                return 1.0 / d if d != 0 else -4000  # type: ignore

            def dihedral(p0, p1, p2, p3):
                """
                Return dihedral angle of points p0,p1,p2,p3
                p0,p1,p2,p3 are atoms positions in cartesian coordinates
                """
                # Praxeolitic formula
                # 1 sqrt, 1 cross product
                # FROM: https://stackoverflow.com/questions/20305272/
                if (
                    (p0 == p1).all()
                    or (p0 == p2).all()
                    or (p0 == p3).all()
                    or (p1 == p2).all()
                    or (p1 == p3).all()
                    or (p2 == p3).all()
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
                return elements_func(X, "Z")  # type: ignore
            elif k == 1:
                if X is None or atoms_list is None:
                    return print("Error! No atoms list or atoms to count")
                else:
                    return atom_counts(X, atoms_list)
            elif k == 2:
                if d == 0.0:
                    return -4000
                else:
                    return 1.0 / d  # type: ignore
            elif k == 3:
                return angles(p[0], p[1], p[2], d[0], d[1])  # type: ignore
            elif k == 4:
                return dihedral(p[0], p[1], p[2], p[3])  # type: ignore
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
                        for ll in var1:
                            trial.append(i + " " + j + " " + k + " " + ll)
            trial = list(np.core.defchararray.split(trial, " "))  # type: ignore
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
                return array

            npad = [(0, 0) for x in range(axis_nb)]
            npad[axis] = (0, pad_size)

            b = np.pad(array, pad_width=npad, mode="constant", constant_values=0)

            return b

        # informations for fast indexing
        atoms = np.array(structure["atom"], dtype="str")
        # elements = np.unique(atoms)
        Natoms = len(atoms)
        atm_lst = range(len(atoms))

        # get all positions for fast indexing
        # compute distances matrix
        if np.greater(k_idx, 1).any():
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
                if scale:
                    f0 = sc(f0, axis=0, with_mean=False)

                f.append(np.reshape(f0, (1, np.shape(f0)[0])))  # type: ignore
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
                for zz in atoms_iter:  # type: ignore
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
                if scale:
                    f1 = sc(
                        f1.reshape(len(atoms_iter), len(xx1)),  # type: ignore
                        axis=1,
                        with_mean=False,  # type: ignore
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
                for zz1, zz2 in atoms_iter:  # type: ignore
                    temp = []
                    temp2 = []
                    for atm1 in atm_lst:
                        for atm2 in atm_lst:
                            if cutoff is not None:
                                if dd[atm2, atm1] > cutoff:  # type: ignore
                                    temp.append(0.0)
                                    temp2.append(0.0)
                                else:
                                    temp.append(g_k(k=k, d=dd[atm2, atm1]))  # type: ignore
                                    temp2.append(
                                        1
                                        if corr(zz1, atoms[atm1])  # type: ignore
                                        * corr(zz2, atoms[atm2])  # type: ignore
                                        == 1
                                        else 0
                                    )
                            else:
                                temp.append(g_k(k=k, d=dd[atm2, atm1]))  # type: ignore
                                temp2.append(
                                    1
                                    if corr(zz1, atoms[atm1]) * corr(zz2, atoms[atm2])  # type: ignore
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
                if scale:
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

                if sparse:
                    # computing correlation and g_k arrays
                    ii = 0
                    n1 = []
                    n2 = []
                    temp = []
                    temp2 = []
                    for zz1, zz2, zz3 in atoms_iter:  # type: ignore
                        ij = 0
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    if cutoff is not None:
                                        if (
                                            dd[atm3, atm1] > cutoff  # type: ignore
                                            or dd[atm3, atm2] > cutoff  # type: ignore
                                        ):
                                            pass
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[pos[atm3], pos[atm2], pos[atm1]],  # type: ignore
                                                    d=[dd[atm3, atm2], dd[atm3, atm1]],  # type: ignore
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])  # type: ignore
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
                                                p=[pos[atm3], pos[atm2], pos[atm1]],  # type: ignore
                                                d=[dd[atm3, atm2], dd[atm3, atm1]],  # type: ignore
                                            )
                                        )
                                        temp2.append(
                                            1
                                            if corr(zz1, atoms[atm1])  # type: ignore
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
                    for i in range(len(atoms_iter)):  # type: ignore
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
                    for zz1, zz2, zz3 in atoms_iter:  # type: ignore
                        temp = []
                        temp2 = []
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    if cutoff is not None:
                                        if (
                                            dd[atm3, atm1] > cutoff  # type: ignore
                                            or dd[atm3, atm2] > cutoff  # type: ignore
                                        ):
                                            temp.append(0.0)
                                            temp2.append(0.0)
                                        else:
                                            temp.append(
                                                g_k(
                                                    k=k,
                                                    p=[pos[atm3], pos[atm2], pos[atm1]],  # type: ignore
                                                    d=[dd[atm3, atm2], dd[atm3, atm1]],  # type: ignore
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])  # type: ignore
                                                * corr(zz2, atoms[atm2])
                                                * corr(zz3, atoms[atm3])
                                                == 1
                                                else 0
                                            )
                                    else:
                                        temp.append(
                                            g_k(
                                                k=k,
                                                p=[pos[atm3], pos[atm2], pos[atm1]],  # type: ignore
                                                d=[dd[atm3, atm2], dd[atm3, atm1]],  # type: ignore
                                            )
                                        )
                                        temp2.append(
                                            1
                                            if corr(zz1, atoms[atm1])  # type: ignore
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
                    for i in range(len(atoms_iter)):  # type: ignore
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
                if scale:
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

                if sparse:
                    # computing correlation and g_k arrays
                    ii = 0
                    n1 = []
                    n2 = []
                    temp = []
                    temp2 = []
                    for zz1, zz2, zz3, zz4 in atoms_iter:  # type: ignore
                        ij = 0
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    for atm4 in atm_lst:
                                        dis = np.array(
                                            [
                                                dd[atm1, atm2],  # type: ignore
                                                dd[atm1, atm3],  # type: ignore
                                                dd[atm1, atm4],  # type: ignore
                                                dd[atm2, atm3],  # type: ignore
                                                dd[atm2, atm4],  # type: ignore
                                                dd[atm3, atm4],  # type: ignore
                                            ]
                                        )
                                        if cutoff is not None:
                                            if np.any(dis > cutoff):
                                                pass
                                            else:
                                                temp.append(
                                                    g_k(
                                                        k=k,
                                                        p=[
                                                            pos[atm1],  # type: ignore
                                                            pos[atm2],  # type: ignore
                                                            pos[atm3],  # type: ignore
                                                            pos[atm4],  # type: ignore
                                                        ],
                                                    )
                                                )
                                                temp2.append(
                                                    1
                                                    if corr(zz1, atoms[atm1])  # type: ignore
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
                                                        pos[atm1],  # type: ignore
                                                        pos[atm2],  # type: ignore
                                                        pos[atm3],  # type: ignore
                                                        pos[atm4],  # type: ignore
                                                    ],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])  # type: ignore
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
                    for i in range(len(atoms_iter)):  # type: ignore
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
                    for zz1, zz2, zz3, zz4 in atoms_iter:  # type: ignore
                        temp = []
                        temp2 = []
                        for atm1 in atm_lst:
                            for atm2 in atm_lst:
                                for atm3 in atm_lst:
                                    for atm4 in atm_lst:
                                        dis = np.array(
                                            [
                                                dd[atm1, atm2],  # type: ignore
                                                dd[atm1, atm3],  # type: ignore
                                                dd[atm1, atm4],  # type: ignore
                                                dd[atm2, atm3],  # type: ignore
                                                dd[atm2, atm4],  # type: ignore
                                                dd[atm3, atm4],  # type: ignore
                                            ]
                                        )
                                        if cutoff is not None:
                                            if np.any(dis > cutoff):
                                                temp.append(0.0)
                                                temp2.append(0.0)
                                            else:
                                                temp.append(
                                                    g_k(
                                                        k=k,
                                                        p=[
                                                            pos[atm1],  # type: ignore
                                                            pos[atm2],  # type: ignore
                                                            pos[atm3],  # type: ignore
                                                            pos[atm4],  # type: ignore
                                                        ],
                                                    )
                                                )
                                                temp2.append(
                                                    1
                                                    if corr(zz1, atoms[atm1])  # type: ignore
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
                                                        pos[atm1],  # type: ignore
                                                        pos[atm2],  # type: ignore
                                                        pos[atm3],  # type: ignore
                                                        pos[atm4],  # type: ignore
                                                    ],
                                                )
                                            )
                                            temp2.append(
                                                1
                                                if corr(zz1, atoms[atm1])  # type: ignore
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
                if scale:
                    f4 = sc(f4, axis=1, with_mean=False)
                f.append(f4)
                xx.append(xx4)
                elements_it.append(atoms_iter)
            idx += 1

        # final mbtr
        f = np.array(f, dtype=object)
        f_pad = []
        if pad:
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
