from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from abc import abstractmethod

import logging
import logging.config
from string import digits
from typing import List
import numpy as np
import os
import re
import sys
import subprocess as sp

import constants as co
from math_util import measure_angle, measure_bond

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


def remove_none(*args):
    return [x for x in args if (x is not None and x != "")]


class Datum(object):
    """
    Class for a reference or calculated data point. TODO
    """

    __slots__ = [
        "_lbl",
        "val",
        "wht",
        "typ",
        "com",
        "src_1",
        "src_2",
        "idx_1",
        "idx_2",
        "atm_1",
        "atm_2",
        "atm_3",
        "atm_4",
        "ff_row",
    ]

    def __init__(
        self,
        lbl=None,
        val=None,
        wht=None,
        typ=None,
        com=None,
        src_1=None,
        src_2=None,
        idx_1=None,
        idx_2=None,
        atm_1=None,
        atm_2=None,
        atm_3=None,
        atm_4=None,
        ff_row=None,
    ):
        self._lbl = lbl
        self.val = val
        self.wht = wht
        self.typ = typ
        self.com = com
        self.src_1 = src_1
        self.src_2 = src_2
        self.idx_1 = idx_1
        self.idx_2 = idx_2
        self.atm_1 = atm_1
        self.atm_2 = atm_2
        self.atm_3 = atm_3
        self.atm_4 = atm_4
        self.ff_row = ff_row

    def __repr__(self):
        return "{}({:7.4f})".format(self.lbl, self.val)

    @property
    def lbl(self):
        if self._lbl is None:
            a = self.typ
            if self.src_1:
                b = re.split("[.]+", self.src_1)[0]
            # Why would it ever not have src_1?
            else:
                b = None
            c = "-".join([str(x) for x in remove_none(self.idx_1, self.idx_2)])
            d = "-".join(
                [
                    str(x)
                    for x in remove_none(self.atm_1, self.atm_2, self.atm_3, self.atm_4)
                ]
            )
            abcd = remove_none(a, b, c, d)
            self._lbl = "_".join(abcd)
        return self._lbl


def datum_sort_key(datum):
    """
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    """
    return (datum.typ, datum.src_1, datum.src_2, datum.idx_1, datum.idx_2)


class Atom(object):
    """
    Data class for a single atom.
    """

    __slots__ = [
        "atom_type",
        "atom_type_name",
        "atomic_num",
        "atomic_mass",
        "bonded_atom_indices",
        "coords_type",
        "_element",
        "_exact_mass",
        "index",
        "partial_charge",
        "x",
        "y",
        "z",
        "props",
    ]

    def __init__(
        self,
        atom_type: str = None,
        atom_type_name: str = None,
        atomic_num: int = None,
        atomic_mass: float = None,
        bonded_atom_indices=None,
        coords=None,
        coords_type=None,
        element: str = None,
        exact_mass=None,
        index: int = None,
        partial_charge: float = None,
        x: float = None,
        y: float = None,
        z: float = None,
    ):
        """Atom object containing relevant properties and metadata.

        Units: Angstrom

        Note:
            TODO Values are all optional because of the currently established q2mm code, however this
            is bad practice, any strictly necessary properties should be required arguments, such as atom type, index, and
            coordinates.

        Args:
            atom_type (str, optional): The integer atom type according to atom.typ file. Defaults to None.
            atom_type_name (str, optional): The name of the atom type corresponding to the integer atom type in the atom.typ file. Defaults to None.
            atomic_num (int, optional): Atomic number (element number) of the atom. Defaults to None.
            atomic_mass (float, optional): TODO. Defaults to None.
            bonded_atom_indices (TODO, optional): TODO. Defaults to None.
            coords (TODO maybe np.ndarray, optional): Atom coordinates. Defaults to None.
            coords_type (TODO, optional): TODO Is this even ever used?. Defaults to None.
            element (str, optional): The atom element (e.g. C, N, O, H). Defaults to None.
            exact_mass (_type_, optional): TODO. Defaults to None.
            index (int, optional): The index number of the atom in its original structural file. Defaults to None.
            partial_charge (float, optional): TODO is this even really used?. Defaults to None.
            x (float, optional): X coordinate of the atom. Defaults to None.
            y (float, optional): Y coordinate of the atom. Defaults to None.
            z (float, optional): Z coordinate of the atom. Defaults to None.
        """
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num  # This is the atom index in the original structure file, 1-based NOT 0-based
        self.atomic_mass = atomic_mass
        self.bonded_atom_indices = bonded_atom_indices
        self.coords_type = coords_type
        self._element = element
        self._exact_mass = exact_mass
        self.index = index
        self.partial_charge = partial_charge
        self.x = x
        self.y = y
        self.z = z
        if coords:  # coordinates are all in Angstroms and Cartesian
            self.x = float(coords[0])
            self.y = float(coords[1])
            self.z = float(coords[2])
        self.props = {}

    def __repr__(self):
        return "{}[{},{},{}]".format(self.atom_type_name, self.x, self.y, self.z)

    @property
    def coords(self) -> np.ndarray:
        """Getter method for coords property.

        Returns:
            np.ndarray: Array of Cartesian coordinates of atom of form [x, y, z].
        """
        return np.array([self.x, self.y, self.z])

    @coords.setter
    def coords(self, value):
        """Setter method for coords property.

        Args:
            value (TODO): Cartesian coordinates of atom of form [x, y, z]
        """
        try:
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        except TypeError:
            pass

    @property
    def element(self):
        if self._element is None:
            self._element = co.MASSES.items()[self.atomic_num - 1][0]
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def exact_mass(self):
        if self._exact_mass is None:
            self._exact_mass = co.MASSES[self.element]
        return self._exact_mass

    @exact_mass.setter
    def exact_mass(self, value):
        self._exact_mass = value

    @property
    def is_dummy(self):
        """
        Return True if self is a dummy atom, else return False.

        Returns
        -------
        bool
        """
        # I think 61 is the default dummy atom type in a Schrodinger atom.typ
        # file.
        # Okay, so maybe it's not. Anyway, Tony added an atom type 205 for
        # dummies. It'd be really great if we all used the same atom.typ file
        # someday.
        # Could add in a check for the atom_type number. I removed it.
        if self.atom_type_name == "Du" or self.element == "X" or self.atomic_num == -2:
            return True
        else:
            return False


class DOF(object):
    """
    Abstract data class for a single degree of freedom.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
    ):
        """Abstract Class for a Degree Of Freedom (DOF) containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the DOF. Defaults to None.
            comment (str, optional): Any comment associated with the DOF TODO Is this used?. Defaults to None.
            value (float, optional): Value of the DOF. Defaults to None.
            ff_row (int, optional): Row of the FF which models the DOF. Defaults to None.
        """
        self.atom_nums: List[int] = atom_nums
        """ TODO atom_indices is a more intuitive name, 
        but use of this property is too widespread (with poor referencing) to change atm,
        refactor this name when there is time."""
        self.comment = comment
        self.value = value
        self.ff_row = ff_row

    def __repr__(self):
        return "{}[{}]({})".format(
            self.__class__.__name__, "-".join(map(str, self.atom_nums)), self.value
        )

    def as_data(self, **kwargs):
        # Sort of silly to have all this stuff about angles and
        # torsions in here, but they both inherit from this class.
        # I suppose it'd make more sense to create a structural
        # element class that these all inherit from.
        # Warning that I recently changed these labels, and that
        # may have consequences.
        if self.__class__.__name__.lower() == "bond":
            typ = "b"
        elif self.__class__.__name__.lower() == "angle":
            typ = "a"
        elif self.__class__.__name__.lower() == "torsion":
            typ = "t"
        datum = Datum(val=self.value, typ=typ, ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, "atm_{}".format(i + 1), atom_num)
        for k, v in kwargs.items():
            setattr(datum, k, v)
        return datum

    def is_same_DOF(self, other) -> bool:
        """Comparison operator for DOFs. Returns true if the DOFs are identical
        based on the indices of the atoms involved. Relies on the assumption that
        atom indices are not different from structure to structure, TODO this is a
        fallacy which should be addressed at some point or at least emphasized to
        the user in documentation.

        Args:
            other (DOF): The DOF to which to compare self.

        Returns:
            bool: True if DOF is identical to self, else False.
        """
        assert other is DOF
        return all(self.atom_nums == other.atom_nums) or all(
            reversed(self.atom_nums) == other.atom_nums
        )


class Bond(DOF):
    """
    Data class for a single bond.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row", "order"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
        order: str = None,
    ):
        """Bond object containing the bare bones properties necessary.

        Note:
            As of yet, there is no need for this class to also track a force constant as it
            is used exclusively in the newer, schrodinger-independent code like seminario.py,
            but this could be a good place to store that data within a FF object if Param were
            to be replaced/condensed.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the bond of the form [index1, index2]. Defaults to None.
            comment (str, optional): Any comment associated with the Bond TODO Is this used?. Defaults to None.
            value (float, optional): Bond length in Angstrom. Defaults to None.
            ff_row (int, optional): Row of the FF which models the bond. Defaults to None.
            order (int, optional): Bond order (e.g. single bond - 1, double bond - 2...)
        """
        super(Bond, self).__init__(atom_nums, comment, value, ff_row)
        self.order = order


class Angle(DOF):
    """
    Data class for a single angle.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Angle object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the Angle. Defaults to None.
            comment (str, optional): Any comment associated with the Angle TODO Is this used?. Defaults to None.
            value (float, optional): Value of the Angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Angle. Defaults to None.
        """
        super(Angle, self).__init__(atom_nums, comment, value, ff_row)


class Torsion(DOF):
    """
    Data class for a single torsion.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Torsion/Dihedral object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the torsion. Defaults to None.
            comment (str, optional): Any comment associated with the torsion TODO Is this used?. Defaults to None.
            value (float, optional): Value of the torsion angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Torsion. Defaults to None.
        """
        super(Torsion, self).__init__(atom_nums, comment, value, ff_row)


class Structure(object):
    """
    Data for a single structure/conformer/snapshot.
    """

    __slots__ = [
        "_atoms",
        "_bonds",
        "_angles",
        "_torsions",
        "hess",
        "props",
        "origin_name",
    ]

    def __init__(self, origin_name: str):
        # TODO: This should really be a constructor which accepts a bare minimum of
        # these fields and the rest are optional defaulted to None, good for error-protection
        # and just generally cleaner, more intuitive. An empty structure is never itself used,
        # so why have it as an option which simply complicates error-checking and tracking.
        self._atoms: List[Atom] = None
        self._bonds: List[Bond] = None
        self._angles: List[Angle] = None
        self._torsions: List[Torsion] = None
        self.hess = None
        self.props = {}
        self.origin_name: str = origin_name

    @property
    def coords(self):
        """
        Returns atomic coordinates as a list of lists.
        """
        return [atom.coords for atom in self._atoms]

    @property
    def num_atoms(self):
        if self._atoms is None or self._atoms == []:
            return self.guess_atoms()
        else:
            return len(self.atoms)

    @property
    def atoms(self):
        # if self._atoms == []:
        #     raise Exception(
        #         "structure._atoms is not defined, this must be done on creation."
        #     )
        if self._atoms is None:
            self._atoms: List[Atom] = []
        return self._atoms

    @property
    def bonds(self):
        # if self._bonds == []:
        #     raise Exception(
        #         "structure._bonds is not defined, this must be done on creation."
        #     )
        if self._bonds is None:
            self._bonds: List[Bond] = []
        return self._bonds

    @property
    def angles(self):
        # if self._angles == []:
        #     self._angles = self.identify_angles() TODO move this to Mol2.structures None property if
        if self._angles is None:
            self._angles: List[Angle] = []
        return self._angles

    @property
    def torsions(self):
        # if self._torsions == []:
        #     self._torsions = self.identify_torsions()
        if self._torsions is None:
            self._torsions: List[Torsion] = []
        return self._torsions

    def generalize_to_ff_atom_types(
        self, equivalency_dict: dict, substr_atom_types: list
    ):
        for atom in self.atoms:
            if (
                atom.atom_type_name not in substr_atom_types
                and atom.atom_type_name in equivalency_dict.keys()
            ):
                atom.atom_type_name = equivalency_dict[atom.atom_type_name]

    def guess_atoms(self) -> int:
        max_atom_index = 0
        for bond in self.bonds:
            max_in_bond = np.max(bond.atom_nums)
            if max_in_bond > max_atom_index:
                max_atom_index = max_in_bond
        return max_atom_index

    # region Methods which ought to be refactored or might be unused but I'm too busy/scared to mess with yet

    def format_coords(self, format="latex", indices_use_charge=None):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.

        latex  - Makes a LaTeX table.
        gauss  - Makes output that matches Gaussian's .com filse.
        jaguar - Just like Gaussian, but include the atom number after the
                 element name in the left column.
        """
        # Formatted for LaTeX.
        if format == "latex":
            output = [
                "\\begin{tabular}{l S[table-format=3.6] "
                "S[table-format=3.6] S[table-format=3.6]}"
            ]
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                output.append(
                    "{0}{1} & {2:3.6f} & {3:3.6f} & "
                    "{4:3.6f}\\\\".format(ele, i + 1, atom.x, atom.y, atom.z)
                )
            output.append("\\end{tabular}")
            return output
        # Formatted for Gaussian .com's.
        elif format == "gauss":
            output = []
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                if indices_use_charge:
                    if atom.index in indices_use_charge:
                        output.append(
                            " {0:s}--{1:.5f}{2:>16.6f}{3:16.6f}"
                            "{4:16.6f}".format(
                                ele, atom.partial_charge, atom.x, atom.y, atom.z
                            )
                        )
                    else:
                        output.append(
                            " {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}".format(
                                ele, atom.x, atom.y, atom.z
                            )
                        )
                else:
                    output.append(
                        " {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}".format(
                            ele, atom.x, atom.y, atom.z
                        )
                    )
            return output

    def select_stuff(self, typ, com_match=None):
        """
        A much simpler version of select_data. It would be nice if select_data
        was a wrapper around this function.
        """
        stuff = []
        for thing in getattr(self, typ):
            if (
                com_match and any(x in thing.comment for x in com_match)
            ) or com_match is None:
                stuff.append(thing)
        return stuff

    def select_data(self, typ, com_match=None, **kwargs):
        """
        Selects bonds, angles, or torsions from the structure and returns them
        in the format used as data.

        typ       - 'bonds', 'angles', or 'torsions'.
        com_match - String or None. If None, just returns all of the selected
                    stuff (bonds, angles, or torsions). If a string, selects
                    only those that have this string in their comment.

                    In .mmo files, the comment corresponds to the substructures
                    name. This way, we only fit bonds, angles, and torsions that
                    directly depend on our parameters.
        """
        data = []
        logger.log(1, ">>> typ: {}".format(typ))
        for thing in getattr(self, typ):
            if (
                com_match and any(x in thing.comment for x in com_match)
            ) or com_match is None:
                datum = thing.as_data(**kwargs)
                # If it's a torsion we have problems.
                # Have to check whether an angle inside the torsion is near 0 or 180.
                if typ == "torsions":
                    atom_nums = [datum.atm_1, datum.atm_2, datum.atm_3, datum.atm_4]
                    angle_atoms_1 = [atom_nums[0], atom_nums[1], atom_nums[2]]
                    angle_atoms_2 = [atom_nums[1], atom_nums[2], atom_nums[3]]
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_1):
                            angle_1 = angle.value
                            break
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_2):
                            angle_2 = angle.value
                            break
                    try:
                        logger.log(1, ">>> atom_nums: {}".format(atom_nums))
                        logger.log(
                            1, ">>> angle_1: {} / angle_2: {}".format(angle_1, angle_2)
                        )
                    except UnboundLocalError:
                        logger.error(">>> atom_nums: {}".format(atom_nums))
                        logger.error(">>> angle_atoms_1: {}".format(angle_atoms_1))
                        logger.error(">>> angle_atoms_2: {}".format(angle_atoms_2))
                        if "angle_1" not in locals():
                            logger.error("Can't identify angle_1!")
                        else:
                            logger.error(">>> angle_1: {}".format(angle_1))
                        if "angle_2" not in locals():
                            logger.error("Can't identify angle_2!")
                        else:
                            logger.error(">>> angle_2: {}".format(angle_2))
                        logger.warning("WARNING: Using torsion anyway!")
                        data.append(datum)
                    if (
                        -20.0 < angle_1 < 20.0
                        or 160.0 < angle_1 < 200.0
                        or -20.0 < angle_2 < 20.0
                        or 160.0 < angle_2 < 200.0
                    ):
                        logger.log(
                            1, ">>> angle_1 or angle_2 is too close to 0 or 180!"
                        )
                        pass
                    else:
                        data.append(datum)
                    # atom_coords = [x.coords for x in atoms]
                    # tor_1 = geo_from_points(
                    #     atom_coords[0], atom_coords[1], atom_coords[2])
                    # tor_2 = geo_from_points(
                    #     atom_coords[1], atom_coords[2], atom_coords[3])
                    # logger.log(1, '>>> tor_1: {} / tor_2: {}'.format(
                    #     tor_1, tor_2))
                    # if -5. < tor_1 < 5. or 175. < tor_1 < 185. or \
                    #         -5. < tor_2 < 5. or 175. < tor_2 < 185.:
                    #     logger.log(
                    #         1,
                    #         '>>> tor_1 or tor_2 is too close to 0 or 180!')
                    #     pass
                    # else:
                    #     data.append(datum)
                else:
                    data.append(datum)
        assert data, "No data actually retrieved!"
        return data

    def get_hyds(self):
        """
        Returns the atom numbers of all hydrogens.

        This might be MM3-specific, but could be reused for QFUERZA TODO: MF
        """
        hyds = []
        for atom in self._atoms:
            if atom.element == "H":
                for bonded_atom_index in atom.bonded_atom_indices:
                    hyds.append(atom)
        logger.log(5, "  -- {} hydrogen(s).".format(len(hyds)))
        return hyds

    def get_dummy_atom_indices(self):
        """
        Returns a list of integers where each integer corresponds to an atom
        that is a dummy atom.

        Returns
        -------
        list of integers
        """
        dummies = []
        for atom in self._atoms:
            if atom.is_dummy:
                logger.log(10, "  -- Identified {} as a dummy atom.".format(atom))
                dummies.append(atom.index)
        return dummies

    # endregion

    def identify_angles(self) -> List[Angle]:
        """Returns angles identified and measured within self Structure.

        Note:
            TODO May need to add same logic of 0 vs 180 as in filetypes.py

        Returns:
            List[Angle]: angles in self Structure
        """
        angles: List[Angle] = []
        i = 0
        for a in self.bonds:
            i += 1
            for b in self.bonds[i:]:
                a1_index, a2_index = a.atom_nums
                b1_index, b2_index = b.atom_nums
                if a1_index == b1_index:
                    if a2_index != b2_index:
                        angle = measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a2_index, a1_index, b2_index], value=angle)
                        )
                if a1_index == b2_index:
                    if a2_index != b1_index:
                        angle = measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a2_index, a1_index, b1_index], value=angle)
                        )
                if a2_index == b2_index:
                    if a1_index != b1_index:
                        angle = measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a1_index, a2_index, b1_index], value=angle)
                        )
                if a2_index == b1_index:
                    if a1_index != b2_index:
                        angle = measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a1_index, a2_index, b2_index], value=angle)
                        )
        return angles

    def identify_torsions(self):  # TODO
        raise NotImplemented

    def get_atoms_in_DOF(self, dof: DOF) -> List[Atom]:
        """Returns a list of Atom objects which are involved in the DOF as implied by atom indices.

        Args:
            dof (DOF): Degree of Freedom (Bond, Angle, etc.) to query

        Returns:
            List[Atom]: Atom objects involved in the DOF.
        """
        return [self.atoms[idx - 1] for idx in dof.atom_nums]

    def get_DOF_atom_types_dict(self) -> dict:
        """Returns a dictionary of the atom types which correspond to each DOF in self.

        Returns:
            dict: dictionary of the form {DOF: [atom_type_name1, atom_type_name2, ...]}
        """
        dof_atom_type_dict = dict()
        for bond in self.bonds:
            dof_atom_type_dict[bond] = [
                atom.atom_type_name for atom in self.get_atoms_in_DOF(bond)
            ]
        for angle in self.angles:
            dof_atom_type_dict[angle] = [
                atom.atom_type_name for atom in self.get_atoms_in_DOF(angle)
            ]
        return dof_atom_type_dict

    def get_eqbm_geom_values(self):
        """
        Gather bonds and angles from structures. Adapted from parameters.py code.

        Ex.:
          bond_dic = {1857: [2.2233, 2.2156, 2.5123],
                      1858: [1.3601, 1.3535, 1.3532]
                     }
        """

        bond_dic = {}
        angle_dic = {}
        torsion_dic = {}
        for bond in self.bonds:
            if bond.ff_row in bond_dic:
                bond_dic[bond.ff_row].append(bond.value)
            else:
                bond_dic[bond.ff_row] = [bond.value]
        for angle in self.angles:
            if angle.ff_row in angle_dic:
                angle_dic[angle.ff_row].append(angle.value)
            else:
                angle_dic[angle.ff_row] = [angle.value]
        for torsion in self.torsions:
            if torsion.ff_row in torsion_dic:
                torsion_dic[torsion.ff_row].append(torsion.value)
            else:
                torsion_dic[torsion.ff_row] = [torsion.value]
        return bond_dic, angle_dic, torsion_dic


class ParamError(Exception):
    pass


class ParamFE(Exception):
    pass


class ParamBE(Exception):
    pass


class Param(object):
    """
     A single parameter of a force field (FF). TODO rework this to match Google style docstrings
     for later sphinx autodocumentation.

     :var _allowed_range: Stored as None if not set, else it's set to True or
       False depending on :func:`allowed_range`.
    :type _allowed_range: None, 'both', 'pos', 'neg'

     :ivar ptype: Parameter type can be one of the following: ae, af, be, bf, df,
       imp1, imp2, sb, or q.
     :type ptype: string

     Attributes
     ----------
     d1 : float
          First derivative of parameter with respect to penalty function.
     d2 : float
          Second derivative of parameter with respect to penalty function.
     step : float
            Step size used during numerical differentiation.
     ptype : {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}
     value : float
             Value of the parameter.
    """

    __slots__ = ["_allowed_range", "_step", "_value", "d1", "d2", "ptype", "simp_var"]

    def __init__(
        self, d1: float = None, d2: float = None, ptype=None, value: float = None
    ):
        """_summary_

        Args:
            d1 (float, optional): First derivative of parameter with respect to penalty function. Defaults to None.
            d2 (float, optional): Second derivative of parameter with respect to penalty function. Defaults to None.
            ptype (_type_, optional): Parameter type {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}. Defaults to None.
            value (float, optional): Value of the parameter. Defaults to None.
        """
        self._allowed_range = None
        self._step = None
        self._value = None
        self.d1 = d1
        self.d2 = d2
        self.ptype = ptype
        self.simp_var = None
        self.value = value

    def __repr__(self):
        return "{}[{}]({:7.4f})".format(self.__class__.__name__, self.ptype, self.value)

    @property
    def allowed_range(self) -> List[float]:
        """Returns the allowed range of values for the parameter based on its parameter type (ptype).

        Returns:
            List[float]: [minimum_value, maximum_value]
        """
        if self._allowed_range is None and self.ptype is not None:
            if self.ptype in ["q", "df"]:
                self._allowed_range = [-float("inf"), float("inf")]
            else:
                self._allowed_range = [0.0, float("inf")]
        return self._allowed_range

    @property
    def step(self):
        """TODO Google style
        Returns a float for the current step size that should be used. If
        _step is a string, return float(_step) * value. If
        _step is a float, simply return that.

        Not sure how well the check for a step size of zero works.
        """
        if self._step is None:
            try:
                self._step = co.STEPS[self.ptype]
            except KeyError:
                logger.warning(
                    "{} doesn't have a default step size and none "
                    "provided!".format(self)
                )
                raise
        if sys.version_info > (3, 0):
            if isinstance(self._step, str):
                return float(self._step) * self.value
            else:
                return self._step
        else:
            if isinstance(self._step, basestring):
                return float(self._step) * self.value
            else:
                return self._step

    @step.setter
    def step(self, x):
        self._step = x

    @property
    def value(self):
        if self.ptype == "ae" and self._value > 180.0:
            self._value = 180.0 - abs(180 - self._value)
        return self._value

    @value.setter
    def value(self, value):
        """TODO Google style
        When you try to give the parameter a value, make sure that's okay.
        """
        if self.value_in_range(value):
            self._value = value

    def convert_and_set(self, value: float, units=None):
        """Converts force constant value in kJ/molA to the correct units based on FF units and parameter type.

        Note: This should only be used for force constants, not equilibrium bond lengths or angles or charges.

        Args:
            value (float): New value for the parameter
            units (str, optional): units to convert to for FF, must be in constants.py. Defaults to None.
        """
        if value is None:
            return
        if units == co.MM3FF:
            self.value = (
                value / co.MM3_STR
            )  #  Uses the conversion factor specific to MM3.fld, Notes on this in box TODO: Remove in a later commit and note commit # in documentation
            # self.value = value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG**2)  if self.ptype == 'bf' else value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG)
            # self.value = value * co.AU_TO_MDYNA  if self.ptype == 'bf' else value * co.AU_TO_MDYN_ANGLE
            # self.value = value * 10**6  if self.ptype == 'bf' else value * co.KJMOLA_TO_MDYN
            # self.value = (
            #     value / co.MDYNA_TO_KJMOLA2
            #     if self.ptype == "bf"
            #     else value * co.KJMOLA_TO_MDYN
            # )
        elif (
            units == co.AMBERFF
        ):  # TODO Amber conversion factor is unknown, ask David Case because it is not just units.
            # self.value = (
            #     value * co.HARTREE_TO_KCALMOL / (co.BOHR_TO_ANG**2)
            #     if self.ptype == "bf"
            #     else value * co.HARTREE_TO_KCALMOL
            # )
            self.value = value * co.HARTREE_TO_KCALMOL / co.HARTREE_TO_KJMOL
        elif units == co.TINKERFF:
            raise NotImplemented()
        else:
            raise Exception(
                "Only MM3, AMBER, and Tinker type force fields have defined units and conversions for parameters in Q2MM."
            )

    def value_in_range(self, value):
        """TODO

        Args:
            value (_type_): _description_

        Raises:
            ParamBE: _description_
            ParamFE: _description_
            ParamError: _description_

        Returns:
            _type_: _description_
        """
        if self.allowed_range[0] <= value <= self.allowed_range[1]:
            return True
        elif value == self.allowed_range[0] - 0.1:
            raise ParamBE(
                "{} Backward Error. Forward Derivative only".format(str(self))
            )
        elif value == self.allowed_range[1] + 0.1:
            raise ParamFE(
                "{} Forward Error. Backward Derivative only".format(str(self))
            )
        elif value == self.allowed_range[1] or value == self.allowed_range[0]:
            return True
        else:
            raise ParamError(
                "{} isn't allowed to have a value of {}! "
                "({} <= x <= {})".format(
                    str(self), value, self.allowed_range[0], self.allowed_range[1]
                )
            )

    def value_at_limits(self):
        """TODO"""
        # Checks if the parameter is at the limits of
        # its allowed range. Should only be run at the
        # end of an optimization to warn users they should
        # consider whether this is ok.
        if self.value == min(self.allowed_range):
            logger.warning(
                "{} is equal to its lower limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self), self.value
                )
            )
        if self.value == max(self.allowed_range):
            logger.warning(
                "{} is equal to its upper limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self), self.value
                )
            )


class FF(object):
    """TODO
    Class for any type of force field.

    path   - Self explanatory.
    data   - List of Datum objects.
    method - String describing method used to generate this FF.
    params - List of Param objects.
    score  - Float which is the objective function score.
    """

    __slots__ = ["path", "data", "method", "params", "score"]

    def __init__(
        self, path=None, data=None, method=None, params: List[Param] = None, score=None
    ):
        self.path = path
        self.data = data
        self.method = method
        self.params: List[Param] = params
        self.score = score

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        ff : `datatypes.FF`
        """
        ff.path = self.path

    def __repr__(self):
        return "{}[{}]({})".format(self.__class__.__name__, self.method, self.score)

    @abstractmethod
    def get_DOFs_by_param(self, structs: List[Structure]) -> dict:
        raise NotImplemented


# region AMBER


class ParAMBER(Param):
    """
    Adds information to Param that is specific to AMBER parameters. TODO
    """

    __slots__ = ["atom_labels", "atom_types", "ff_col", "ff_row", "mm3_label"]

    def __init__(
        self,
        atom_labels=None,
        atom_types=None,
        ff_col=None,
        ff_row=None,
        mm3_label=None,
        d1=None,
        d2=None,
        ptype=None,
        value=None,
    ):
        self.atom_labels = atom_labels
        self.atom_types = [atom_type.strip() for atom_type in atom_types]
        self.ff_col = ff_col
        self.ff_row = ff_row
        self.mm3_label = mm3_label
        super(ParAMBER, self).__init__(ptype=ptype, value=value)

    def __repr__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.ff_row, self.ff_col, self.value
        )

    def __str__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.ff_row, self.ff_col, self.value
        )

    def convert_and_set(self, value):
        return super().convert_and_set(value, units=co.AMBERFF)


class AmberFF(FF):
    """
    STUFF TO FILL IN LATER TODO
    """

    units = co.AMBERFF

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        super(AmberFF, self).__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None
        # change constant
        co.STEPS["bf"] = 10.00
        co.STEPS["af"] = 10.0
        co.STEPS["df"] = 10.0

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        """
        ff.path = self.path
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def import_ff(self, path=None, sub_search="OPT"):
        if path is None:
            path = self.path
        bonds = ["bond", "bond3", "bond4", "bond5"]
        pibonds = ["pibond", "pibond3", "pibond4", "pibond5"]
        angles = ["angle", "angle3", "angle4", "angle5"]
        torsions = ["torsion", "torsion4", "torsion5"]
        dipoles = ["dipole", "dipole3", "dipole4", "dipole5"]
        self.params: List[Param] = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
        count = 0
        with open(path, "r") as f:
            logger.log(15, "READING: {}".format(path))
            for i, line in enumerate(f):
                split = line.split()
                if not q2mm_sec and "# Q2MM" in line:
                    q2mm_sec = True
                elif q2mm_sec and "#" in line[0]:
                    self.sub_names.append(line[1:])
                    if "OPT" in line:
                        gather_data = True
                    else:
                        gather_data = False
                if gather_data and split:
                    if "MASS" in line and count == 0:
                        count = 1
                        continue
                    if "BOND" in line and count == 1:
                        count = 2
                        continue
                    elif count == 1 and "ANGL" not in line:
                        # atom symbol:atomic mass:atomic polarizability
                        at = split[0]  # need number if it matters
                        el = split[0]
                        mass = split[1]
                        if len(split) > 2:
                            pol = split[2]
                        # no need for atom label
                        # at = ["Z0", "P1", "CX"]
                    # BOND
                    if "ANGL" in line and count == 2:
                        count = 3
                        continue
                    elif count == 2 and "DIHE" not in line:
                        # A1-A2 Force Const in kcal/mol/(A**2): Eq. length in A
                        AA = line[:5].split("-")
                        BB = line[5:].split()
                        at = [AA[0], AA[1]]
                        self.params.extend(
                            (
                                ParAMBER(
                                    atom_types=at,
                                    ptype="bf",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                ParAMBER(
                                    atom_types=at,
                                    ptype="be",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # ANGLE
                    if "DIHE" in line and count == 3:
                        count = 4
                        continue
                    elif count == 3 and "IMPR" not in line:
                        AA = line[: 2 + 3 * 2].split("-")
                        BB = line[2 + 3 * 2 :].split()
                        at = [AA[0], AA[1], AA[2]]
                        self.params.extend(
                            (
                                ParAMBER(
                                    atom_types=at,
                                    ptype="af",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                ParAMBER(
                                    atom_types=at,
                                    ptype="ae",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # Dihedral
                    if "IMPR" in line and count == 4:
                        count = 5
                        continue
                    elif count == 4 and "NONB" not in line:
                        # (PK/IDIVF) * (1 + cos(PN*phi - PHASE))
                        # A4 IDIVF PK PHASE PN
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="df",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(BB[1]),
                            )
                        )

                    # Improper
                    if "NONB" in line and count == 5:
                        count = 6
                        continue
                    elif count == 5:
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="imp1",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(BB[0]),
                            )
                        )

                    #                    # Hbond
                    #                    if "NONB" in line and count == 6:
                    #                        count == 7
                    #                        continue
                    #                    elif count == 6:
                    #                        0

                    # NONB
                    if count == 6:
                        continue

                    if "vdw" == split[0]:
                        # The first float is the vdw radius, the second has to do
                        # with homoatomic well depths and the last is a reduction
                        # factor for univalent atoms (I don't think we will need
                        # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="vdw",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(split[2]),
                            )
                        )
        logger.log(15, "  -- Read {} parameters.".format(len(self.params)))

    def export_ff(self, path=None, params: List[ParAMBER] = None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params: List[Param] = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, ">>> param: {} param.value: {}".format(param, param.value))
            line = lines[param.ff_row - 1]
            if abs(param.value) > 1999.0:
                logger.warning("Value of {} is too high! Skipping write.".format(param))
            else:
                atoms = ""
                const = ""
                space3 = " " * 3
                col = int(param.ff_col - 1)
                value = "{:7.4f}".format(param.value)
                tempsplit = line.split("-")
                leng = len(tempsplit)
                AA = None
                BB = None
                if leng == 2:
                    # Bond
                    nl = 2 + 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 5
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng == 3:
                    # Angle
                    nl = 2 + 3 * 2
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 4
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng >= 4:
                    # Dihedral/Improper
                    nl = 2 + 3 * 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 2
                    value = "{:7.5f}".format(param.value)
                    if param.ptype == "imp1":
                        atoms += space3
                        BB[0] = value
                        const = (
                            "".join([format(el, ">12") for el in BB[:3]])
                            + space3
                            + " ".join(BB[3:])
                        )
                    else:
                        atoms += format(BB[0], ">3")
                        # Dihedral
                        BB[1] = value
                        const = (
                            "".join([format(el, ">12") for el in BB[1:4]])
                            + space3
                            + " ".join(BB[4:])
                        )

                lines[param.ff_row - 1] = atoms + const + "\n"
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, "WROTE: {}".format(path))

    def get_DOFs_by_atom_type(self, structs: List[Structure]) -> dict:
        dof_by_param = dict()
        for param in self.params:
            dof_by_param[param.ff_row]: List[DOF] = ([])  
            # TODO: this ^ should be fine bc it will always have a ff_row, but at some point consider just making ff_row a universal Param property, I don't see why not.
        for struct in structs:
            for bond in struct.bonds:
                dof_by_param[bond.ff_row].append(bond)
            for angle in struct.angles:
                dof_by_param[angle.ff_row].append(angle)
            for dihed in struct.torsions:
                dof_by_param[dihed.ff_row].append(dihed)
        return dof_by_param

    def get_DOFs_by_param(self, structs: List[Structure]) -> dict:
        return self.get_DOFs_by_atom_type(structs)


# endregion AMBER

# endregion
