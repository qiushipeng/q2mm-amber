#!/usr/bin/env python
"""
Contains basic utility methods for use in Q2MM.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import copy
import numpy as np
import logging
from logging import config
from typing import List
#import parmed

import constants as co
from data_structs import *

config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

#Note: the file i/o utility methods here will make the relevant data objects, imported from data_structs
#TODO: MF this ^ might need to be re-evaluated when considering file write/output, but could be handled
# by making a file i/o object from here in the loop/calculate or whatever module and having an object initialization
# method which takes in the data_structs object which is relevant. then it's still a one-sided dependency

# region Atom Type Handling

#TODO: MF finish documenting this code

def convert_atom_type(atom_type: str) -> str:
    """_summary_

    Args:
        atom_type (str): _description_

    Returns:
        str: _description_
    """
    q2mm_atom_type = "".join(filter(str.isalnum, atom_type))
    q2mm_atom_type = q2mm_atom_type.upper()
    # TODO: MF Add a check to verify it is included in atom.typ here,
    # exception should be caught, propagated, and handled here to avoid
    # silent failure within MacroModel upon FF export (or other silent or loud failures).
    return q2mm_atom_type


def convert_atom_type_pair(atom_type_pair):
    q2mm_atom_type_pair = [convert_atom_type(atom_type) for atom_type in atom_type_pair]
    return q2mm_atom_type_pair


def convert_atom_types(atom_type_pairs: list) -> list:
    q2mm_atom_type_pairs = [
        convert_atom_type_pair(atom_type_pair) for atom_type_pair in atom_type_pairs
    ]
    return q2mm_atom_type_pairs


def is_same_type_DOF(atom_types1: list, atom_types2: list) -> bool:
    reverse_1 = copy.deepcopy(atom_types1)
    reverse_1.reverse()
    return atom_types1 == atom_types2 or reverse_1 == atom_types2


# endregion Atom Type Handling

#region Mass Weighting
def mass_weight_hessian(hess, atoms, reverse=False):
    """Mass weights Hessian by multiplying my 1/sqrt(mass1 * mass2). If reverse is True,
     it un-mass weights the Hessian. Note that this does not return a new object but rather
     modifies the one passed as hess.

    Args:
        hess (_type_): Hessian matrix to mass-weight, modifies the variable itself.
        atoms (_type_): Atom objects related to the Hessian (must be in correct order).
        reverse (bool, optional): Whether to reverse mass-weight (* sqrt(mass1 * mass2)). Defaults to False.
    """
    masses = [co.MASSES[x.element] for x in atoms if not x.is_dummy]
    changes = []
    for mass in masses:
        changes.extend([1 / np.sqrt(mass)] * 3)
    x, y = hess.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                hess[i, j] = hess[i, j] / changes[i] / changes[j]
            else:
                hess[i, j] = hess[i, j] * changes[i] * changes[j]


def mass_weight_force_constant(
    force_const: float, atoms: List[Atom], reverse: bool = False, rm: bool = False
) -> float:
    """Mass weights force constant. If reverse is True, it un-mass weights
    the force constant.

    Args:
        force_const (float): force constant value to mass-weight or un-mass-weight.
        atoms (List[Atom]): Atoms associated with the force constant.
        reverse (bool, optional): Whether to un-mass-weight the force constant instead. Defaults to False.
        rm (bool, optional): Whether to instead convert the force constant to reduced mass representation. Defaults to False.

    Returns:
        float: mass-weighted or un-mass-weighted value of force constant.
    """
    force_constant = force_const
    masses = [co.MASSES[x.element] for x in atoms]
    changes = []
    if rm:
        return force_constant * np.sqrt(masses[0] + masses[1])
    for mass in masses:
        change = 1 / np.sqrt(mass)
        if reverse:
            force_constant = force_constant / change
        else:
            force_constant = force_constant * change
    return force_constant


def mass_weight_eigenvectors(evecs, atoms, reverse=False):
    """
    Mass weights eigenvectors. If reverse is True, it un-mass weights
    the eigenvectors. TODO
    """
    changes = []
    for atom in atoms:
        if not atom.is_dummy:
            changes.extend([np.sqrt(atom.exact_mass)] * 3)
    x, y = evecs.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                evecs[i, j] /= changes[j]
            else:
                evecs[i, j] *= changes[j]

#endregion Mass Weighting

#TODO: MF all references to the following method should be replaced by calls to linear_algebra.invert_ts_curvature
def replace_minimum(array, value=1):
    """
    Replace the minimum vallue in an arbitrary NumPy array. Historically,
    the replace value is either 1 or co.HESSIAN_CONVERSION.
    """
    minimum = array.min()
    minimum_index = np.where(array == minimum)
    assert minimum < 0, 'Minimum of array is not negative!'
    # It would be better to address this in a different way. This particular
    # data structure just isn't what we want.
    array.setflags(write=True)
    logger.log(logging.INFO,"max eval: "+str(array.max()))
    # Sometimes we use 1, but sometimes we use co.HESSIAN_CONVERSION.
    array[minimum_index] = value
    logger.log(logging.DEBUG, '>>> minimum_index: {}'.format(minimum_index))
    logger.log(logging.DEBUG, '>>> array:\n{}'.format(array))
    logger.log(logging.INFO, '  -- Replaced minimum in array with {}.'.format(value))

#region File I/O

class File(object):
    """
    Base for every other filetype class. Identical to filetypes.py version,
    ported over for schrodinger independence in seminario.py
    """

    __slots__ = ["_lines", "path", "directory", "filename"]

    def __init__(self, path: str):
        """Instantiates a file object fro the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): location of the file
        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]

    @property
    def lines(self) -> List[str]:
        """Returns the lines of the file.

        Returns:
            List[str]: lines of the file
        """
        if self._lines is None:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path, lines=None):
        """Writes lines to file at path.

        Args:
            path (str): location of file to write
            lines (List[str], optional): lines to write to file. Defaults to None, which then writes self.lines.
        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)

class Mol2(File):
    """
    Used to retrieve structural data from mol2 files.

    Please ensure that mol2 atom types match the atom types specified in the force field.

    Note:
            Format for the data in the file can be found by searching
            Tripos Mol2 File Format SYBYL.
    """

    TRIPOS_FLAG = "@<TRIPOS>"
    MOLECULE_FLAG = "MOLECULE"
    ATOM_FLAG = "ATOM"
    BOND_FLAG = "BOND"

    __slots__ = ["_lines", "path", "directory", "filename", "_structures"]

    def __init__(self, path: str):
        """Creates a Mol2 object based on the path given, data is only structural.

        Args:
            path (str): Absolute path of mol2 file.
        """
        super(Mol2, self).__init__(path)
        self._structures: List[Structure] = None
        #TODO: MF - In general, refactor all of these such that None is instead an empty list or string, not None

    @property
    def structures(self) -> List[Structure]:
        """Returns the Structure objects extracted from the mol2 file at self.path.
        If None, indicating no extraction yet, parses the lines from the file to populate
        the structures list with Structures.

        Returns:
            List[Structure]: Structure objects extracted from parsing the mol2 file.
        """
        if self._structures is None:
            self.parse_lines()
        return self._structures

    def parse_lines(self):
        """Parses self.lines() as set by super to extract Structure objects to self.structures.

        It is safe to parse this with split because the mol2 format from SYBYL
         requires consistent data ordering matching the standard, otherwise the
         file is not in valid mol2 format.
        """
        # TODO this could be amended to use regular expression matching (regex) if slow
        self._structures: List[Structure] = []
        joined_lines = "".join(self.lines)
        structure_chunks = joined_lines.split(self.TRIPOS_FLAG + self.MOLECULE_FLAG)
        entry_num = 0 if len(structure_chunks) > 2 else None
        for struct_chunk in structure_chunks:
            if struct_chunk != "":
                self._structures.append(self.parse_structure(struct_chunk, chunk_index=entry_num))

        if len(structure_chunks) - 1 != len(self._structures):
            logger.log(
                logging.WARN,
                "Only "
                + str(len(self._structures))
                + " structures could be parsed from "
                + str(len(structure_chunks) - 1)
                + " MOLECULE entries in the .mol2 file",
            )

    def parse_atoms(self, atom_lines: List[str]) -> List[Atom]:
        """Returns the Atom objects parsed from the atom_lines given.

        Args:
            atom_lines (List[str]): lines from the mol2 file pertaining to the atoms in the structure.

        Returns:
            List[Atom]: Atom objects parsed from atom_lines
        """
        atoms = []
        for atom_entry in atom_lines:
            if atom_entry == "" or atom_entry.strip() == self.ATOM_FLAG:
                continue
            atom_split = atom_entry.split()
            for chr in atom_split[1]:
                if chr.isdigit():
                    numer_index = atom_split[1].index(chr)
                    break
            element = atom_split[1][0:numer_index] if numer_index else atom_split[1]
            atoms.append(
                Atom(
                    index=int(atom_split[0]),
                    element=element,
                    coords=atom_split[2:5],
                    atom_type_name=atom_split[5],
                    partial_charge=float(atom_split[8]),
                )
            )
        return atoms

    def parse_bonds(self, bond_lines: List[str], structure: Structure) -> List[Bond]:
        """Returns the Bond objects parsed from the bond_lines given.

        Args:
            bond_lines (List[str]): lines from the mol2 file pertaining to the bond connectivity in the structure.
            structure (Structure): structure which the bonds pertain to, used for bond measurement.

        Returns:
            List[Bond]: Bond objects parsed from bond_lines
        """
        bonds = []
        for bond_entry in bond_lines:
            if bond_entry == "" or bond_entry.strip() == self.BOND_FLAG:
                continue
            bond_split = bond_entry.split()
            a_index = int(bond_split[1])
            b_index = int(bond_split[2])
            bonds.append(
                Bond(
                    atom_nums=[a_index, b_index],
                    order=bond_split[3],
                    value= measure_bond(
                        structure.atoms[a_index - 1].coords,
                        structure.atoms[b_index - 1].coords,
                    ),
                )
            )

        # TODO: Ideally, the bonds class would measure the bonds and just contain a pointer to an Atom
        # object, but that would require a decent-sized refactor so hold off for now
        # TODO: MF Consider implementing this now that this is a stand-alone AMBER-specific refactor

        return bonds

    def parse_structure(self, structure_chunk: str, chunk_index:int = None) -> Structure:
        """Returns the Structure objects parsed from the structure_chunk given.

        Args:
            structure_chunk (str): string containing the lines which pertain to a single structure.

        Returns:
            Structure: the Structure object parsed from structure_chunk data.
        """
        tripos_chunks = structure_chunk.split(self.TRIPOS_FLAG)
        molecule_lines = tripos_chunks[0].split("\n")
        atom_lines = tripos_chunks[1].split("\n")
        bond_chunk = 2
        bond_lines = tripos_chunks[bond_chunk].split("\n")

        # assert that data was chunked correctly:
        assert atom_lines[0].strip() == self.ATOM_FLAG
        while bond_lines[0].strip() != self.BOND_FLAG:
            bond_chunk += 1
            try:
                bond_lines = tripos_chunks[bond_chunk].split("\n")
            except IndexError:
                logger.log(
                    logging.ERROR,
                    "No BOND flag within mol2 MOLECULE, invalid structure.",
                )

        # parse number of atoms and number of bonds from line 2 below @<TRIPOS>MOLECULE
        molecule_data = molecule_lines[2].split()
        num_atoms = int(molecule_data[0])
        num_bonds = int(molecule_data[1])

        file_identifier = self.filename if chunk_index is None else self.filename + str(chunk_index)

        struct = (
            Structure(file_identifier)
        )  # ideally we would gather data, then instantiate a Structure with
        # all the data as arguments, but for now I will follow the precedent within the Q2MM code
        # to avoid significant refactoring since we still don't have test cases or test scripts
        #TODO: MF We can now do this ^ ! because it will be gathered by the file io in utilities, 
        # then we create a Structure object with all of that information. This refactoring is a great task
        # to divvy up between QP and MF

        # send chunk from @<TRIPOS>ATOM to @<TRIPOS>BOND to parse_atoms
        struct._atoms = self.parse_atoms(atom_lines)

        # use num atoms from @<TRIPOS>MOLECULE to verify parse is correct
        assert (
            len(struct._atoms) == num_atoms
        ), "Parsed {} atoms but only expected {} atoms based on Mol2 data.".format(
            len(struct.atoms), num_atoms
        )
        assert all(
            struct.atoms[i].index == i + 1 for i in range(len(struct.atoms))
        ), "Mol2 atom index values do not match their ordering."

        # send chunk from @<TRIPOS>BOND to end-of-file to parse_bonds
        struct._bonds = self.parse_bonds(bond_lines, struct)

        # use num bonds from @<TRIPOS>MOLECULE to verify parse is correct
        assert (
            len(struct._bonds) == num_bonds
        ), "Parsed {} bonds but only expected {} bonds based on Mol2 data.".format(
            len(struct.bonds), num_bonds
        )

        struct._angles =  struct.identify_angles()

        return struct

    def value_bonds(
        self,
    ):  # TODO Not currently in use, remove if not needed by March 1, 2024.
        atom_list = self._structures.atoms
        for bond in self._structures.bonds:
            # Indexing atom_list is possible only because this is within the Mol2 class so
            # we can assume that the atoms were added in order of their atom index.
            atom1 = atom_list[bond.atom_nums[0] - 1]
            atom2 = atom_list[bond.atom_nums[1] - 1]
            bond.value = measure_bond(
                np.array(atom1.coords), np.array(atom2.coords)
            )

#region Reference Files
class GaussLog(File):
    """
    Retrieves data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """

    __slots__ = [
        "_lines",
        "path",
        "directory",
        "filename",
        "_evals",
        "_evecs",
        "_structures",
        "_esp_rms",
        "_au_hessian",
    ]

    def __init__(self, path: str, au_hessian=False):
        """Instantiates a file object for the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): location of the Gaussian log file
            au_hessian (bool, optional): If true, Hessian will not be converted to
            kJ/(mol*Angstrom^2) but rather left in Atomic Units (AU) (Hartree/Bohr^2).
            Defaults to False.
        """
        super(GaussLog, self).__init__(path)
        self._evals = None
        self._evecs = None
        self._structures = None
        self._esp_rms = None
        self._au_hessian = au_hessian

    @property
    def evecs(self):
        """Returns eigenvectors of frequency analysis if applicable.  If not yet parsed,
        parses them from the log body, not the archive.

        Returns:
            evecs (np.array): eigenvectors of Gaussian frequency analysis
        """
        if self._evecs is None:
            self.read_out()
        return self._evecs

    @property
    def evals(self):
        """Returns eigenvalues of frequency analysis if applicable.  If not yet parsed,
        parses them from the log body, not the archive.

        Returns:
            evals (np.array) : eigenvalues of Gaussian frequency analysis
        """
        if self._evals is None:
            self.read_out()
        return self._evals

    @property
    def structures(self) -> List[Structure]:
        """Returns Structure objects parsed from the Gaussian log file. If None,
        parses the archive of the log file for structures.

        Returns:
            List[Structure]: Structures parsed from log file archive.
        """
        if self._structures is None:
            # self.read_out()
            self.read_archive()
        return self._structures

    @property
    def esp_rms(self):
        """Returns the esp_rms (Electrostatic potential ?? TODO)

        Returns:
            int | float: TODO
        """
        if self._esp_rms is None:
            self._esp_rms = -1
            self.read_out()
        return self._esp_rms

    def read_out(self):
        """
        Read force constant and eigenvector data from a frequency
        calculation.
        """
        logger.log(5, "READING: {}".format(self.filename))
        self._evals = []
        self._evecs = []
        self._structures = []
        force_constants = []
        evecs = []
        with open(self.path, "r") as f:
            # The keyword "harmonic" shows up before the section we're
            # interested in. It can show up multiple times depending on the
            # options in the Gaussian .com file.
            past_first_harm = False
            # High precision mode, turned on by including "freq=hpmodes" in the
            # Gaussian .com file.
            hpmodes = False
            file_iterator = iter(f)
            # This while loop breaks when the end of the file is reached, or
            # if the high quality modes have been read already.
            while True:
                try:
                    line = next(file_iterator)
                except:
                    # End of file.
                    break
                if "Charges from ESP fit" in line:
                    pattern = re.compile("RMS=\s+({0})".format(co.RE_FLOAT))
                    match = pattern.search(line)
                    self._esp_rms = float(match.group(1))
                # Gathering some geometric information.
                elif "Standard orientation:" in line:
                    self._structures.append(Structure(self.filename))
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while not "---" in line:
                        cols = line.split()
                        self._structures[-1].atoms.append(
                            Atom(
                                index=int(cols[0]),
                                atomic_num=int(cols[1]),
                                x=float(cols[3]),
                                y=float(cols[4]),
                                z=float(cols[5]),
                            )
                        )
                        line = next(file_iterator)
                    logger.log(
                        5,
                        "  -- Found {} atoms.".format(len(self._structures[-1].atoms)),
                    )
                elif "Harmonic" in line:
                    # The high quality eigenvectors come before the low quality
                    # ones. If you see "Harmonic" again, it means you're at the
                    # low quality ones now, so break.
                    if past_first_harm:
                        break
                    else:
                        past_first_harm = True
                elif "Frequencies" in line:
                    # We're going to keep reusing these.
                    # We accumulate sets of eigevectors and eigenvalues, add
                    # them to self._evecs and self._evals, and then reuse this
                    # for the next set.
                    del force_constants[:]
                    del evecs[:]
                    # Values inside line look like:
                    #     "Frequencies --- xxxx.xxxx xxxx.xxxx"
                    # That's why we remove the 1st two columns. This is
                    # consistent with and without "hpmodes".
                    # For "hpmodes" option, there are 5 of these frequencies.
                    # Without "hpmodes", there are 3.
                    # Thus the eigenvectors and eigenvalues will come in sets of
                    # either 5 or 3.
                    cols = line.split()
                    for frequency in map(float, cols[2:]):
                        # Has 1. or -1. depending on the sign of the frequency.
                        if frequency < 0.0:
                            force_constants.append(-1.0)
                        else:
                            force_constants.append(1.0)
                        # For now this is empty, but we will add to it soon.
                        evecs.append([])

                    # Moving on to the reduced masses.
                    line = next(file_iterator)
                    cols = line.split()
                    # Again, trim the "Reduced masses ---".
                    # It's "Red. masses --" for without "hpmodes".
                    for i, mass in enumerate(map(float, cols[3:])):
                        # +/- 1 / reduced mass
                        force_constants[i] = force_constants[i] / mass

                    # Now we are on the line with the force constants.
                    line = next(file_iterator)
                    cols = line.split()
                    # Trim "Force constants ---". It's "Frc consts --" without
                    # "hpmodes".
                    for i, force_constant in enumerate(map(float, cols[3:])):
                        # co.AU_TO_MDYNA = 15.569141
                        force_constants[i] *= force_constant / co.AU_TO_MDYNA

                    # Force constants were calculated above as follows:
                    #    a = +/- 1 depending on the sign of the frequency
                    #    b = a / reduced mass (obtained from the Gaussian log)
                    #    c = b * force constant / conversion factor (force
                    #         (constant obtained from Gaussian log) (conversion
                    #         factor is inside constants module)

                    # Skip the IR intensities.
                    next(file_iterator)
                    # This is different depending on whether you use "hpmodes".
                    line = next(file_iterator)
                    # "Coord" seems to only appear when the "hpmodes" is used.
                    if "Coord" in line:
                        hpmodes = True
                    # This is different depending on whether you use
                    # "freq=projected".
                    line = next(file_iterator)
                    # The "projected" keyword seems to add "IRC Coupling".
                    if "IRC Coupling" in line:
                        line = next(file_iterator)
                    # We're on to the eigenvectors.
                    # Until the end of this section containing the eigenvectors,
                    # the number of columns remains constant. When that changes,
                    # we know we're to the next set of frequencies, force
                    # constants and eigenvectors.
                    # Actually check that we've moved on, sometimes a "Depolar" entry is
                    if "Depolar" in line:
                        line = next(file_iterator)
                    if "Atom" in line:
                        line = next(file_iterator)
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        # This will come after all the eigenvectors have been
                        # read. We can break out then.
                        if "Harmonic" in line:
                            break
                        # If "hpmodes" is used, you have an extra column here
                        # that is simply an index.
                        if hpmodes:
                            cols = cols[1:]
                        # cols corresponds to line(s) (maybe only 1st line)
                        # under section "Coord Atom Element:" (at least for
                        # "hpmodes").

                        # Just the square root of the mass from co.MASSES.
                        # co.MASSES currently has the average mass.
                        # Gaussian may use the mass of the most abundant
                        # isotope. This may be a problem.
                        mass_sqrt = np.sqrt(
                            list(co.MASSES.items())[int(cols[1]) - 1][1]
                        )

                        cols = cols[2:]
                        # This corresponds to the same line still, but without
                        # the atom elements.

                        # This loop expands the LoL, evecs, as so.
                        # Iteration 1:
                        # [[x], [x], [x], [x], [x]]
                        # Iteration 2:
                        # [[x, x], [x, x], [x, x], [x, x], [x, x]]
                        # ... etc. until the length of the sublist is equal to
                        # the number of atoms. Remember, for low precision
                        # eigenvectors it only adds in sets of 3, not 5.

                        # Elements of evecs are simply the data under
                        # "Coord Atom Element" multiplied by the square root
                        # of the weight.
                        for i in range(len(evecs)):
                            if hpmodes:
                                # evecs is a LoL. Length of sublist is
                                # equal to # of columns in section "Coord Atom
                                # Element" minus 3, for the 1st 3 columns
                                # (index, atom index, atomic number).
                                evecs[i].append(float(cols[i]) * mass_sqrt)
                            else:
                                # This is fow low precision eigenvectors. It's a
                                # funny way to go in sets of 3. Take a look at
                                # your low precision Gaussian log and it will
                                # make more sense.
                                for useless in range(3):
                                    x = float(cols.pop(0))
                                    evecs[i].append(x * mass_sqrt)
                        line = next(file_iterator)
                        cols = line.split()

                    # Here the overall number of eigenvalues and eigenvectors is
                    # increased by 5 (high precision) or 3 (low precision). The
                    # total number goes to 3N - 6 for non-linear and 3N - 5 for
                    # linear. Same goes for self._evecs.
                    for i in range(len(evecs)):
                        self._evals.append(force_constants[i])
                        self._evecs.append(evecs[i])
                    # We know we're done if this is in the line.
                    if "Harmonic" in line:
                        break
        if self._evals and self._evecs:
            for evec in self._evecs:
                # Each evec is a single eigenvector.
                # Add up the sum of squares over an eigenvector.
                sum_of_squares = 0.0
                # Appropriately named, element is an element of that single
                # eigenvector.
                for element in evec:
                    sum_of_squares += element * element
                # Now x is the inverse of the square root of the sum of squares
                # for an individual eigenvector.
                element = 1 / np.sqrt(sum_of_squares)
                for i in range(len(evec)):
                    evec[i] *= element
            self._evals = np.array(self._evals)
            self._evecs = np.array(self._evecs)
            logger.log(logging.DEBUG, ">>> self._evals: {}".format(self._evals))
            logger.log(logging.DEBUG, ">>> self._evecs: {}".format(self._evecs))
            logger.log(5, "  -- {} structures found.".format(len(self.structures)))

    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self):
        """
        Only reads last archive found in the Gaussian .log file. Hessian converted
        to kJ/molA^2
        """
        logger.log(5, "READING: {}".format(self.filename))
        struct = Structure(self.filename)
        self._structures = [struct]
        # Matches everything in between the start and end.
        # (?s)  - Flag for re.compile which says that . matches all.
        # \\\\  - One single \
        # Start - " 1\1\".
        # End   - Some number of \ followed by @. Not sure how many \ there
        #         are, so this matches as many as possible. Also, this could
        #         get separated by a line break (which would also include
        #         adding in a space since that's how Gaussian starts new lines
        #         in the archive).
        # We pull out the last one [-1] in case there are multiple archives
        # in a file.
        #        print(self.path)
        #        print(open(self.path,'r').read())
        #        print(re.findall('(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)',open(self.path,'r').read()))
        try:
            arch = re.findall(
                "(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)", open(self.path, "r").read()
            )[-1]
            logger.log(5, "  -- Located last archive.")
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        # Make it into one string.
        arch = arch.replace("\n ", "")
        # Separate it by Gaussian's section divider.
        arch = arch.split("\\\\")
        # Helps us iterate over sections of the archive.
        section_counter = 0
        # SECTION 0
        # General job information.
        arch_general = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            "\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)"
            "\\\\(?P<date>.*?)"
            "\\\\.*?",
            arch_general,
        )
        struct.props["user"] = stuff.group("user")
        struct.props["date"] = stuff.group("date")
        # SECTION 1
        # The commands you wrote.
        arch_commands = arch[section_counter]
        section_counter += 1
        # SECTION 2
        # The comment line.
        arch_comment = arch[section_counter]
        section_counter += 1
        # SECTION 3
        # Actually has charge, multiplicity and coords.
        arch_coords = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            "(?P<charge>.*?)" ",(?P<multiplicity>.*?)" "\\\\(?P<atoms>.*)", arch_coords
        )
        struct.props["charge"] = stuff.group("charge")
        struct.props["multiplicity"] = stuff.group("multiplicity")
        # We want to do more fancy stuff with the atoms than simply add to
        # the properties dictionary.
        atoms = stuff.group("atoms")
        atoms = atoms.split("\\")
        # Z-matrix coordinates adds another section. We need to be aware of
        # this.
        probably_z_matrix = False
        struct._atoms = []
        for atom in atoms:
            stuff = atom.split(",")
            # An atom typically looks like this:
            #    C,0.1135,0.13135,0.63463
            if len(stuff) == 4:
                ele, x, y, z = stuff
            # But sometimes they look like this (notice the extra zero):
            #    C,0,0.1135,0.13135,0.63463
            # I'm not sure what that extra zero is for. Anyway, ignore
            # that extra whatever if it's there.
            elif len(stuff) == 5:
                ele, x, y, z = stuff[0], stuff[2], stuff[3], stuff[4]
            # And this would be really bad. Haven't seen anything else like
            # this yet.
            # 160613 - So, not sure when I wrote that comment, but something
            # like this definitely happens when using scans and z-matrices.
            # I'm going to ignore grabbing any atoms in this case.
            else:
                logger.warning(
                    "Not sure how to read coordinates from Gaussian acrhive!"
                )
                probably_z_matrix = True
                section_counter += 1
                # Let's have it stop looping over atoms, but not fail anymore.
                break
                # raise Exception(
                #     'Not sure how to read coordinates from Gaussian archive!')
            struct._atoms.append(Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        logger.log(logging.INFO, "  -- Read {} atoms.".format(len(struct._atoms)))
        # SECTION 4
        # All sorts of information here. This area looks like:
        #     prop1=value1\prop2=value2\prop3=value3
        arch_info = arch[section_counter]
        section_counter += 1
        arch_info = arch_info.split("\\")
        for thing in arch_info:
            prop_name, prop_value = thing.split("=")
            struct.props[prop_name] = prop_value
        # SECTION 5
        # The Hessian. Only exists if you did a frequency calculation.
        # Appears in lower triangular form, not mass-weighted.
        if not arch[section_counter] == "@":
            hess_tri = arch[section_counter]
            hess_tri = hess_tri.split(",")
            logger.log(
                5,
                "  -- Read {} Hessian elements in lower triangular "
                "form.".format(len(hess_tri)),
            )
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(5, "  -- Created {} Hessian matrix.".format(hess.shape))
            # Code for if it was in upper triangle (it's not).
            # hess[np.triu_indices_from(hess)] = hess_tri
            # hess += np.triu(hess, -1).T
            # Lower triangle code.
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
            if not self._au_hessian:
                hess *= co.HESSIAN_CONVERSION
            struct.hess = hess

    def get_most_converged(self, structures=None):
        """
        Used with geometry optimizations that don't succeed. Sometimes
        intermediate geometries obtain better convergence than the
        final geometry. This function returns the class Structure for
        the most converged geometry, which can then be used to output
        the coordinates for the next optimization.
        """
        if structures is None:
            structures = self.structures
        structures_compared = 0
        best_structure = None
        best_yes_or_no = None
        fields = [
            "RMS Force",
            "RMS Displacement",
            "Maximum Force",
            "Maximum Displacement",
        ]
        for i, structure in reversed(list(enumerate(structures))):
            yes_or_no = [
                value[2] for key, value in structure.props.items() if key in fields
            ]
            if not structure._atoms:
                logger.warning(
                    "  -- No atoms found in structure {}. " "Skipping.".format(i + 1)
                )
                continue
            if len(yes_or_no) == 4:
                structures_compared += 1
                if best_structure is None:
                    logger.log(logging.DEBUG, "  -- Most converged structure: {}".format(i + 1))
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count("YES") > best_yes_or_no.count("YES"):
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count("YES") == best_yes_or_no.count("YES"):
                    number_better = 0
                    for field in fields:
                        if structure.props[field][0] < best_structure.props[field][0]:
                            number_better += 1
                    if number_better > 2:
                        best_structure = structure
                        best_yes_or_no = yes_or_no
            elif len(yes_or_no) != 0:
                logger.warning(
                    "  -- Partial convergence criterion in structure: {}".format(
                        self.path
                    )
                )
        logger.log(
            10,
            "  -- Compared {} out of {} structures.".format(
                structures_compared, len(self.structures)
            ),
        )
        return best_structure

    def read_optimization(self, coords_type="both"):
        """
        Finds structures from a Gaussian geometry optimization that
        are listed throughout the log file. Also finds data about
        their convergence.

        coords_type = "input" or "standard" or "both"
                      Using both may cause coordinates in one format
                      to be overwritten by whatever comes later in the
                      log file.
        """
        logger.log(logging.DEBUG, "READING: {}".format(self.filename))
        structures = []
        with open(self.path, "r") as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                # Look for start of optimization section of log file and
                # set a flag that it has indeed started.
                if section_optimization and "Optimization stopped." in line:
                    section_optimization = False
                    logger.log(5, "[L{}] End optimization section.".format(i + 1))
                if not section_optimization and "Search for a local minimum." in line:
                    section_optimization = True
                    logger.log(5, "[L{}] Start optimization section.".format(i + 1))
                if section_optimization:
                    # Start of a structure.
                    if "Step number" in line:
                        structures.append(Structure(self.filename))
                        current_structure = structures[-1]
                        logger.log(
                            5,
                            "[L{}] Added structure "
                            "(currently {}).".format(i + 1, len(structures)),
                        )
                    # Look for convergence information related to a single
                    # structure.
                    if section_convergence and "GradGradGrad" in line:
                        section_convergence = False
                        logger.log(5, "[L{}] End convergence section.".format(i + 1))
                    if section_convergence:
                        match = re.match(
                            "\s(Maximum|RMS)\s+(Force|Displacement)\s+({0})\s+"
                            "({0})\s+(YES|NO)".format(co.RE_FLOAT),
                            line,
                        )
                        if match:
                            current_structure.props[
                                "{} {}".format(match.group(1), match.group(2))
                            ] = (
                                float(match.group(3)),
                                float(match.group(4)),
                                match.group(5),
                            )
                    if "Converged?" in line:
                        section_convergence = True
                        logger.log(5, "[L{}] Start convergence section.".format(i + 1))
                    # Look for input coords.
                    if coords_type == "input" or coords_type == "both":
                        # End of input coords for a given structure.
                        if section_coords_input and "Distance matrix" in line:
                            section_coords_input = False
                            logger.log(
                                5,
                                "[L{}] End input coordinates section "
                                "({} atoms).".format(i + 1, count_atom),
                            )
                        # Add atoms and coords to structure.
                        if section_coords_input:
                            match = re.match(
                                "\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+"
                                "({0})".format(co.RE_FLOAT),
                                line,
                            )
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1)) - 1
                                    ]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)
                                    ), (
                                        "[L{}] Atomic numbers don't match "
                                        "(current != existing) "
                                        "({} != {}).".format(
                                            i + 1,
                                            int(match.group(2)),
                                            current_atom.atomic_num,
                                        )
                                    )
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = "input"
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and "Input orientation:" in line:
                            section_coords_input = True
                            count_atom = 0
                            logger.log(
                                5,
                                "[L{}] Start input coordinates "
                                "section.".format(i + 1),
                            )
                    # Look for standard coords.
                    if coords_type == "standard" or coords_type == "both":
                        # End of coordinates for a given structure.
                        if section_coords_standard and (
                            "Rotational constants" in line or "Leave Link" in line
                        ):
                            section_coords_standard = False
                            logger.log(
                                5,
                                "[L{}] End standard coordinates "
                                "section ({} atoms).".format(i + 1, count_atom),
                            )
                        # Grab coords for each atom. Add atoms to the structure.
                        if section_coords_standard:
                            match = re.match(
                                "\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+"
                                "({0})\s+({0})".format(co.RE_FLOAT),
                                line,
                            )
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1)) - 1
                                    ]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)
                                    ), (
                                        "[L{}] Atomic numbers don't match "
                                        "(current != existing) "
                                        "({} != {}).".format(
                                            i + 1,
                                            int(match.group(2)),
                                            current_atom.atomic_num,
                                        )
                                    )
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = "standard"
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of standard coords.
                        if (
                            not section_coords_standard
                            and "Standard orientation" in line
                        ):
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(
                                5,
                                "[L{}] Start standard coordinates "
                                "section.".format(i + 1),
                            )
        return structures

#endregion Reference Files

#region AMBER I/O

class Frcmod(File):
    """
    STUFF TO FILL IN LATER TODO
    """

    units = co.AMBERFF

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        super(Frcmod, self).__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None
        self.force_field:AmberFF = None
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
        self.params: List[ParAMBER] = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
        count = 0
        with open(path, "r") as f:
            logger.log(logging.DEBUG, "READING: {}".format(path))
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
        logger.log(logging.DEBUG, "  -- Read {} parameters.".format(len(self.params)))
        self.ff:AmberFF = AmberFF(self.path, data=None)

    def export_ff(self, path=None, params:List[ParAMBER]=None, lines=None):
        #TODO: MF change this such that it takes in an AmberFF and the AmberFF contains the params, the Frcmod makes/stores the lines
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params:List[ParAMBER] = self.params #TODO: MF - KK what? Unclear why this is obscuring earlier params, 
            # will require close attention when refactoring but should fix whatever this is by refactoring
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(logging.DEBUG, ">>> param: {} param.value: {}".format(param, param.value))
            line = lines[param.ff_row - 1]
            if abs(param.value) > 1999.0:
                logger.warning("Value of {} is too high! Skipping write.".format(param)) #TODO: MF - KK wrote this, no clue why he needed it bc should be using allowed_range
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
        logger.log(logging.DEBUG, "WROTE: {}".format(path))

    def get_DOFs_by_atom_type(self, structs:List[Structure]) -> dict:
        dof_by_param = dict()
        for param in self.params:
            dof_by_param[param.ff_row]:List[DOF] = []
        for struct in structs:
            for bond in struct.bonds:
                dof_by_param[bond.ff_row].append(bond)
            for angle in struct.angles:
                dof_by_param[angle.ff_row].append(angle)
            for dihed in struct.torsions:
                dof_by_param[dihed.ff_row].append(dihed)
        return dof_by_param
    
    def get_DOFs_by_param(self, structs:List[Structure]) -> dict:
        return self.get_DOFs_by_atom_type(structs)
    
class AmberLeapInput(File):
    def __init__(self, path: str, frcmod:Frcmod):
        super(File, self).__init__(path)
        self.frcmod = frcmod

    def write_in_file(self):
        return

# Currently only for 1 system.
# Note: It comes in mass-weighted kcal/mol (A?), then gets converted to kJ/mol but nothing else
class AmberHess(File):
    def __init__(self, path):
        super(AmberHess, self).__init__(path)
        self._hessian = None
        self.natoms = None
    @property
    def hessian(self):
        if self._hessian is None:
            logger.log(logging.DEBUG, 'READING: {}'.format(self.filename))
            with open("./calc/"+self.filename, 'r') as f:
                lines = f.readlines()
            for i,line in enumerate(lines):
                if i == 0:
                    self.natoms = int(line.split()[1])
                    hessian = np.zeros([self.natoms * 3, self.natoms * 3], dtype=float)
                else:
                    row = np.array(line.split()).astype(float)
                    hessian[:,i-1] = row
            # Convert hessian units to use kJ/mol instead of kcal/mol.

            # kcal/mol for energy in AMBER
            # E(kcal/mol -> cm**-1) = 349.75
            # freq = sqrt(lambda(kcal/mol)) / (2 pi c)
            
            w, v = np.linalg.eigh(hessian)
            eigval = np.zeros([self.natoms * 3],dtype=float)
            for i,eig in enumerate(w):
                if eig < 0:
                    eigval[i] = -np.sqrt(-eig)
                else:
                    eigval[i] = np.sqrt(eig)
            eigval *= 108.587 # freq in cm**-1
            self._hessian = hessian / co.HARTREE_TO_KCALMOL \
                * co.HARTREE_TO_KJMOL
            logger.log(5, '  -- Finished Creating {} Hessian matrix.'.format(
                hessian.shape))
            return self._hessian
class AmberEne(File):
    """
        Amber .ene file to read either current energy or optimized energy
    """
    def __init__(self, path):
        super(AmberEne, self).__init__(path)
        self._structures = None
        self.name = None
    @property
    def structures(self):
        if self._structures == None:
            logger.log(logging.DEBUG, 'READING: {}'.format(self.filename))
            self._structures = []
            flag = 0
            with open('./calc/'+self.filename, 'r') as f:
                sections = {'sp':1, 'minimization':2}
                calc_section = 'sp'
                count_previous = 0
                    
                for line in f:
                    count_current = sections[calc_section]
                    if count_current != count_previous:
                        current_structure = Structure(self.filename)
                        self._structures.append(current_structure)
                        count_previous += 1
                    if 'FINAL RESULTS' in line:
                        flag = 1
                    elif flag == 1 and "NSTEP" in line:
                        flag = 2
                    elif flag == 2:
                        energy = self.read_line_for_energy(line)
                        if energy is not None:
                            current_structure.props['energy']=energy
                        flag = 0
            logger.log(5, '  -- Imported {} structure(s)'.format(
                len(self._structures)))
        return self._structures


    def read_line_for_energy(self, line):
        # The Amber Energy is in units of kcal/mol, so we have to convert them to kJ/mol
        # for consistency purposes.
        # don't know how to use match = re.compile 
        linesplit = line.split()
        energy = float(linesplit[1])
        energy *= co.HARTREE_TO_KJMOL / co.HARTREE_TO_KCALMOL
        return energy
#        if match:
#            energy = float(match.group(1))
#            energy *= co.HARTREE_TO_KJMOL / co.HARTREE_TO_KCALMOL
#            return energy
#        else:
#            return None

class AmberGeo(File):
    """
        .geo file to be used for bond,angles,dihedral sets
        .out file for the current value
    """
    def __init__(self, path):
        super(AmberGeo, self).__init__(path)
        self._structures = None
        self.name = None
    @property
    def structures(self):
        if self._structures == None:
            logger.log(logging.DEBUG, 'READING: {}'.format(self.filename))
            self._structures = []
            with open("./calc/"+self.filename, 'r') as f:
                sections = {'sp':1, 'minimization':2, 'hessian':2}
                count_previous = 0
                calc_section = 'sp'
                b = 0
                a = 0  
                t = 0
                for line in f:
                    count_current = sections[calc_section]
                    if count_current != count_previous:
                        bonds = []
                        angles = []
                        torsions = []
                        current_structure = Structure(self.filename)
                        self._structures.append(current_structure)
                        count_previous += 1
                    section = None
                    if "END" in line:
                        t = 0
                        calc_section = 'minimization'
                        for bond in bonds:
                            bond.atom_nums.sort()
                        bonds.sort(key=lambda x: (x.atom_nums[0],
                                                  x.atom_nums[1]))
                        for angle in angles:
                            if angle.atom_nums[0] > angle.atom_nums[2]:
                                angle.atom_nums = [angle.atom_nums[2],
                                                   angle.atom_nums[1],
                                                   angle.atom_nums[0]]
                        for torsion in torsions:
                            if torsion.atom_nums[1] > torsion.atom_nums[2]:
                                torsion.atom_nums = [torsion.atom_nums[3],
                                                     torsion.atom_nums[2],
                                                     torsion.atom_nums[1],
                                                     torsion.atom_nums[0]]
                        angles.sort(key=lambda x: (x.atom_nums[1],
                                                   x.atom_nums[0],
                                                   x.atom_nums[2]))
                        torsions.sort(key=lambda x: (x.atom_nums[1],
                                                     x.atom_nums[2],
                                                     x.atom_nums[0],
                                                     x.atom_nums[3]))
                        current_structure.bonds.extend(bonds)
                        current_structure.angles.extend(angles)
                        current_structure.torsions.extend(torsions)

                    if t == 1:
                        torsion = self.read_line_for_torsion(line)
                        if torsion is not None:
                            torsions.append(torsion)
                    elif 'TORSIONS' in line:
                        t = 1
                        a = 0
                    if a == 1:
                        angle = self.read_line_for_angle(line)
                        if angle is not None:
                            angles.append(angle)
                    elif 'ANGLES' in line:
                        a = 1
                        b = 0
                    if b == 1:
                        bond = self.read_line_for_bond(line)
                        if bond is not None:
                            bonds.append(bond)
                    elif 'BONDS' in line:
                        b = 1


            logger.log(5, '  -- Imported {} structure(s)'.format(
                len(self._structures)))
        return self._structures

    def read_line_for_bond(self, line):
        # All bond data starts with the string "Bond" and then the rest of the
        # interaction information.
        a,b,z = line.split()
        atom_nums = [int(x) for x in [a,b]]
        value = float(z)
        return Bond(atom_nums=atom_nums, value=value)

    def read_line_for_angle(self, line):
        a,b,c,z = line.split()
        atom_nums = [int(x) for x in [a,b,c]]
        value = float(z)
        return Angle(atom_nums=atom_nums, value=value)

    def read_line_for_torsion(self, line):
        a,b,c,d,z = line.split()
        atom_nums = [int(x) for x in [a,b,c,d]]
        value = float(z)
        return Torsion(atom_nums=atom_nums, value=value)

    def read_line_for_energy(self, line):
        # The TPE is in units of kcal/mol, so we have to convert them to kJ/mol
        # for consistency purposes.
        match = re.compile('Total Potential Energy :\s+({0})'.format(
            co.RE_FLOAT)).search(line)
        if match:
            energy = float(match.group(1))
            energy *= co.HARTREE_TO_KJMOL / co.HARTREE_TO_KCALMOL
            return energy
        else:
            return None
class AmberLeap_Gaus(File):
    def __init__(self, path):
        """
            run -> gaus to amber -> sp -> traj -> cpptraj -> cpptraj -> AmberGeo
            path = leap.in
        """
        super(AmberLeap_Gaus, self).__init__(path)
        self._index_output_log = None
        self._structures = None
        self.commands = None
        self.name = os.path.splitext(self.filename)[0]
        self.filename = self.name + '.in' # .log file to .in (.in file is never replaced. so using .in should have original coordinate)
        self.name_log = 'gaus.' + self.name + '.log'
        self.name_prm = 'gaus.' + self.name + '.parm7' #topology
        self.name_rst = 'gaus.' + self.name + '.rst7' # coordinate
        self.name_min = 'gaus.' + self.name + '.min' # sander min input
        self.name_ene = 'gaus.' + self.name + '.ene'
        self.name_dyn = 'gaus.' + self.name + '.dyn' # sander dyn input
        self.name_int = 'gaus.' + self.name + '.int' # interaction input for cpptraj
        self.name_geo = 'gaus.' + self.name + '.geo' # cpptraj output for all interactions (to be read by AmberGeo)
        self.min_script = """Comments
 &cntrl
  imin      = 1,
  ntx       = 1,
  maxcyc    = 0,
  ncyc      = 0,
  cut       = 15.0,
  ntpr      = 10000,
  ntwx      = 0,
  ntb       = 0
 /
"""
        self.dyn_script = """Comments
 &cntrl
  imin      = 0,
  ntx       = 1,
  irest     = 0,
  nstlim    = 0,
  ntwx      = 1,
  cut       = 15.0,
  ntb       = 0,
  ntpr      = 1
 /
"""
    @property
    def structures(self):
        if self._structures is None:
            logger.log(logging.DEBUG, 'READING: {}'.format(self.filename))
            struct = Structure(self.filename)
            self._structures = [struct]
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.split()
                    if len(line) == 2:
                        struct.props['total atoms'] = int(line[0])
                        struct.props['title'] = line[1]
                        logger.log(5, '  -- Read {} atoms.'.format(
                            struct.props['total atoms']))
                    if len(line) > 2:
                        indx, ele, x, y, z, at, bonded_atom = line[0], \
                            line[1], line[2], line[3], line[4], \
                            line[5], line[6:]
                        struct.atoms.append(Atom(index=int(indx),
                            element=ele,
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            atom_type=at,
                            atom_type_name=at,
                            bonded_atom_indices=bonded_atom))
            return self._structures
    def get_com_opts(self):
        com_opts = {'freq': False,
                    'opt': False,
                    'sp':True,
                    'tors': False,
                    'geo':True}
        return com_opts

#BUG: 'fixatomorder' is removed in Himani's version of q2mm_kk, this is correct
# 'fixatomorder' command is removed because it causes mismatches between the line
# numbers of atoms, thus producing nonsensical bond lengths in the output .geo files.
# This was pinpointed by Mikaela and Himani on 11/28/22 and running without this command
# does not crash, produce errors, or result in nonsensical bonds.  It must be removed for
# the gaussian (reference) version of this script as well ~line 597.

    def extract(self,log):
        script="""
trajin calc/gaus.NAME.nc
fixatomorder
AA
run
write
exit
"""
        script = script.replace("NAME",self.name)
        geo = ""

        # read .geo file and store all possible interaction
        bonds = []
        angles = []
        torsions = []
        ref = open('./calc/'+self.name_geo,'r').readlines()
        count = 0
        for line in ref:
            # Bonds
            if "[angles]" in line:
                count = 0
            elif count == 1:
                bonds.append(line.split()[-4:-2])
            elif "Atom2" in line:
                count = 1

            # Angles
            if "[dihedrals]" in line:
                count = 0
            if count == 2:
                angles.append(line.split()[-6:-3])
            elif "Atom3" in line:
                count = 2

            # Dihedral
            # store the columns as negtive since there is unexpected "B" or E in front of column
            if "TIME" in line:
                count = 0
            if count == 3:
                torsions.append(line.split()[-8:-4])
            elif "Atom4" in line:
                count = 3
        
        for a,b in bonds:
            geo += "distance @{} @{} out calc/gaus.bonds".format(a,b) + '\n'
        for a,b,c in angles:
            geo += "angle @{} @{} @{} out calc/gaus.angles".format(a,b,c) + '\n'
        for a,b,c,d in torsions:
            geo += "dihedral @{} @{} @{} @{} out calc/gaus.torsions".format(a,b,c,d) + '\n'
        
        script = script.replace("AA",geo)
        script_f = './calc/' + self.name + '.temp'
        with open(script_f, 'w') as f:
            f.write(script)
        sp.call("cpptraj -p calc/prmtop < {}".format(script_f), shell=True, stderr=log, stdin=log, stdout=log)
        summary = ""
        if os.path.isfile("calc/gaus.bonds"):
            bond_file = open("calc/gaus.bonds","r").readlines()
            bond_line = bond_file[-1].split()[1:]
            summary += "BONDS\n"
            i = 0
            for a,b in bonds:
                summary += "{} {} {} \n".format(a,b,bond_line[i])
                i += 1
        if os.path.isfile("calc/gaus.angles"):
            angle_file = open("calc/gaus.angles","r").readlines()
            angle_line = angle_file[-1].split()[1:]
            summary += "ANGLES\n"
            i = 0
            for a,b,c in angles:
                summary += "{} {} {} {} \n".format(a,b,c,angle_line[i])
                i += 1
        if os.path.isfile("calc/gaus.torsions"):
            tors_file = open("calc/gaus.torsions","r").readlines()
            tors_line = tors_file[-1].split()[1:]
            summary += "TORSIONS\n"
            i = 0
            for a,b,c,d in torsions:
                summary += "{} {} {} {} {} \n".format(a,b,c,d,tors_line[i])
                i += 1
        summary += "END"
        # replace name_geo with summary
        with open('./calc/'+self.name_geo,'w') as f:
            f.write(summary)
        return

    def geometry(self,log):
        # Run Trajectory (Required for cpptraj)
        with open("./calc/"+self.name_dyn, 'w') as f:
            f.write(self.dyn_script)
        sp.call("msander -O -i calc/{} -o calc/traj.out -p calc/prmtop -c calc/gaus.{}.rst -x calc/gaus.{}.nc".format(self.name_dyn,self.name,self.name),shell=True)
        # Generate All geometry
        int_script = "bonds\nangles\ndihedrals\n"
        with open('./calc/'+self.name_int, 'w') as f:
            f.write(int_script)
        sp.call("cpptraj -p calc/prmtop < calc/{} > calc/{} \n".format(self.name_int,self.name_geo),shell=True)
        self.extract(log)
        return
    def run(self,check_tokens=False):
        logger.log(5, 'RUNNING: {}'.format(self.filename))
        self._index_output_log = []
        com_opts = self.get_com_opts()
        current_directory = os.getcwd()
        os.chdir(self.directory)
        log = open(self.name_log,'w')
        os.chdir(self.directory)
        if os.path.isfile('calc'):
            os.remove('calc')
        sp.call("mkdir calc",shell=True, stderr=log, stdin=log, stdout=log)
        if com_opts['sp']:
            logger.log(logging.DEBUG, '  CALCULATE: {}'.format(self.filename))
            # Run leap
            sp.call("tleap -f {}".format(self.filename),shell=True, stderr=log, stdin=log, stdout=log) # parm7 rst7 files made
            # Run Min
            with open("./calc/"+self.name_min, 'w') as f:
                f.write(self.min_script)
            sp.call("msander -O -i calc/{} -o calc/{} -p calc/prmtop -c calc/inpcrd -r calc/gaus.{}.rst".format(self.name_min,self.name_ene,self.name),shell=True, stderr=log, stdin=log, stdout=log)
        if com_opts['geo']:
            self.geometry(log)
        os.chdir(current_directory)

class AmberLeap(File):
    def __init__(self, path):
        """
            path = leap.in
        """
        super(AmberLeap, self).__init__(path)
        self._index_output_log = None
        self._structures = None
        self.commands = None
        self.name = os.path.splitext(self.filename)[0]
        self.name_log = 'amber.' + self.name + '.log'
        self.name_prm = 'amber.' + self.name + '.parm7' #topology
        self.name_rst = 'amber.' + self.name + '.rst7' # coordinate
        self.name_min = 'amber.' + self.name + '.min' # sander min input
        self.name_ene = 'amber.' + self.name + '.ene'
        self.name_dyn = 'amber.' + self.name + '.dyn' # sander dyn input
        self.name_int = 'amber.' + self.name + '.int' # interaction input for cpptraj
        self.name_geo = 'amber.' + self.name + '.geo' # cpptraj output for all interactions
        self.name_hes = 'amber.' + self.name + '.hes'
        self.geo = None
        self.min_script = """Comments
 &cntrl
  imin      = 1,
  ntx       = 1,
  maxcyc    = aa,
  ncyc      = bb,
  cut       = 15.0,
  ntpr      = 10000,
  ntwx      = 0,
  ntb       = 0
 /
"""
        self.dyn_script = """Comments
 &cntrl
  imin      = 0,
  ntx       = 1,
  irest     = 0,
  nstlim    = 0,
  ntwx      = 1,
  cut       = 15.0,
  ntb       = 0,
  ntpr      = 1
 /
"""
    @property
    def structures(self):
        if self._structures is None:
            logger.log(logging.DEBUG, 'READING: {}'.format(self.filename))
            struct = Structure(self.filename)
            self._structures = [struct]
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.split()
                    if len(line) == 2:
                        struct.props['total atoms'] = int(line[0])
                        struct.props['title'] = line[1]
                        logger.log(5, '  -- Read {} atoms.'.format(
                            struct.props['total atoms']))
                    if len(line) > 2:
                        indx, ele, x, y, z, at, bonded_atom = line[0], \
                            line[1], line[2], line[3], line[4], \
                            line[5], line[6:]
                        struct.atoms.append(Atom(index=int(indx),
                            element=ele,
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            atom_type=at,
                            atom_type_name=at,
                            bonded_atom_indices=bonded_atom))
            return self._structures
    def get_com_opts(self):
        com_opts = {'freq': False,
                    'opt': False,
                    'sp': False,
                    'tors': False,
                    'geo':False}
        if any(x in ['ab','aa','at','abo','aao','ato'] for x in self.commands):
            com_opts['geo'] = True
        if any(x in ['abo','aao','ato','aeo','ae1o','aeao'] for x in self.commands):
            com_opts['opt'] = True
            com_opts['sp'] = True
        if any(x in ['ah', 'ajeig', 'ageig'] for x in self.commands):
            com_opts['geo'] = True
            com_opts['freq'] = True
            com_opts['opt'] = True
            com_opts['sp'] = True
        if any(x in ['at', 'ato'] for x in self.commands):
            com_opts['tors'] = True
        return com_opts
    def extract(self,log):
#BUG: 'fixatomorder' is removed in Himani's version of q2mm_kk, this is correct
# 'fixatomorder' command is removed because it causes mismatches between the line
# numbers of atoms, thus producing nonsensical bond lengths in the output .geo files.
# This was pinpointed by Mikaela and Himani on 11/28/22 and running without this command
# does not crash, produce errors, or result in nonsensical bonds.  It must be removed for
# the gaussian (reference) version of this script as well ~line 378.

        script="""
trajin calc/amber.NAME.nc
fixatomorder
AA
run
write
exit
"""
        script = script.replace("NAME",self.name)
        geo = ""

        # read .geo file and store all possible interaction
        bonds = []
        angles = []
        torsions = []
        ref = open('./calc/'+self.name_geo,'r').readlines()
        self.geo = ref
        count = 0
        for line in ref:
            # Bonds
            if "[angles]" in line:
                count = 0
            elif count == 1:
                bonds.append(line.split()[-4:-2])
            elif "Atom2" in line:
                count = 1

            # Angles
            if "[dihedrals]" in line:
                count = 0
            if count == 2:
                angles.append(line.split()[-6:-3])
            elif "Atom3" in line:
                count = 2

            # Dihedral
            # store the columns as negtive since there is unexpected "B" or E in front of column
            if "TIME" in line:
                count = 0
            if count == 3:
                torsions.append(line.split()[-8:-4])
            elif "Atom4" in line:
                count = 3

        for a,b in bonds:
            geo += "distance @{} @{} out calc/amber.bonds".format(a,b) + '\n'
        for a,b,c in angles:
            geo += "angle @{} @{} @{} out calc/amber.angles".format(a,b,c) + '\n'
        for a,b,c,d in torsions:
            geo += "dihedral @{} @{} @{} @{} out calc/amber.torsions".format(a,b,c,d) + '\n'
        script = script.replace("AA",geo)
        script_f = './calc/' + self.name + '.temp'
        with open(script_f, 'w') as f:
            f.write(script)
        sp.call("cpptraj -p calc/prmtop < {}".format(script_f), shell=True, stderr=log, stdin=log, stdout=log)
        summary = ""
        if os.path.isfile("calc/amber.bonds"):
            bond_file = open("calc/amber.bonds","r").readlines()
            bond_line = bond_file[-1].split()[1:]
            summary += "BONDS\n"
            i = 0
            for a,b in bonds:
                summary += "{} {} {} \n".format(a,b,bond_line[i])
                i += 1
        if os.path.isfile("calc/amber.angles"):
            angle_file = open("calc/amber.angles","r").readlines()
            angle_line = angle_file[-1].split()[1:]
            summary += "ANGLES\n"
            i = 0
            for a,b,c in angles:
                summary += "{} {} {} {} \n".format(a,b,c,angle_line[i])
                i += 1
        if os.path.isfile("calc/amber.torsions"):
            tors_file = open("calc/amber.torsions","r").readlines()
            tors_line = tors_file[-1].split()[1:]
            summary += "TORSIONS\n"
            i = 0
            for a,b,c,d in torsions:
                summary += "{} {} {} {} {} \n".format(a,b,c,d,tors_line[i])
                i += 1
        summary += "END"
        # replace name_geo with summary
        with open('./calc/'+self.name_geo,'w') as f:
            f.write(summary)
        return

    def hessian(self,log):
        # if pdb file does not exit, then convert mol2 to pdb
        os.chdir(self.directory)
        if os.path.isfile(self.name+".pdb"):
            0
        else:
            sp.call("antechamber -dr no -i {} -fi mol2 -o {} -fo pdb".format(self.name+".mol2",self.name+".pdb"),shell=True)
        # nab input file
        # dielectric constant = 80.4 for water.
        # currently manual change required
        script = """molecule m;
float x[4000], fret;

m = getpdb("{}.pdb");
readparm(m, "./calc/prmtop");

mm_options( "cut=15., ntpr=1, nsnb=99999, diel = C, dielc = 80.40" );
mme_init( m, NULL, "::Z", x, NULL);
setxyz_from_mol( m, NULL, x );

nmode( x, 3*m.natoms, mme2, 0, 0, 0.0, 0.0, 0);""".format(self.name)
        
#         script = """#include <stdio.h>
# #include <string.h>
# #include <stdlib.h>
# #include <math.h>
# #include <assert.h>
# #include "nabc.h"
# static int mytaskid, numtasks;

# static MOLECULE_T *m;

# static REAL_T x[4000], fret;


# int main( argc, argv )
# 	int	argc;
# 	char	*argv[];
# {
# 	nabout = stdout; /*default*/

# 	mytaskid=0; numtasks=1;
# m = getpdb( "{}.pdb", NULL );
# readparm( m, "./calc/prmtop" );

# mm_options( "cut=15., ntpr=1, nsnb=99999, diel = C, dielc = 80.40" );
# mme_init( m, NULL, "::Z", x, NULL );
# setxyz_from_mol(  &m, NULL, x );

# nmode( x, 3 *  *( NAB_mri( m, "natoms" ) ), mme2, 0, 0, 0.000000E+00, 0.000000E+00, 0 );

# 	exit( 0 );}""".format(self.name)

        with open('./calc/'+self.name+'.nab','w') as f:
            f.write(script)
        # nab compile
        sp.call("nab -v calc/{}.nab -o calc/{}".format(self.name, self.name),shell=True)
        # with open('./calc/'+self.name+'.c','w') as f:
        #     f.write(script)
        # nab compile
        #sp.call("gcc -v calc/{}.c -o calc/{} > gcc.out".format(self.name, self.name),shell=True)
        # nab run
        sp.call("./calc/{}".format(self.name),shell=True,stderr=log, stdin=log, stdout=log)
        # hessian.mat formed
        # rename to .hess
        sp.call("mv ./calc/hessian.mat ./calc/{}".format(self.name_hes),shell = True)
        return
    def geo_extract(self):
    
        bonds = []
        angles = []
        torsions = []
    
        ref = self.geo
        count = 0
        for line in ref:
            # Bonds
            if "[angles]" in line:
                count = 0
            elif count == 1:
                bonds.append(line.split()[-4:-2])
            elif "Atom2" in line:
                count = 1

            # Angles
            if "[dihedrals]" in line:
                count = 0
            if count == 2:
                angles.append(line.split()[-6:-3])
            elif "Atom3" in line:
                count = 2

            # Dihedral
            # store the columns as negtive since there is unexpected "B" or E in front of column
            if "TIME" in line:
                count = 0
            if count == 3:
                torsions.append(line.split()[-8:-4])
            elif "Atom4" in line:
                count = 3

        hes_ele = np.array([None,None,None,None])
        for a,b in bonds:
            hes_ele = np.vstack((hes_ele,[a,b,None,None]))
        for a,b,c in angles:
            hes_ele = np.vstack((hes_ele,[a,b,c,None]))
        for a,b,c,d in torsions:
            hes_ele = np.vstack((hes_ele,[a,b,c,d]))
        np.save("calc/geo",hes_ele)
        return
        
    def geometry(self,log):
        # Run Trajectory (Required for cpptraj)
        with open("./calc/"+self.name_dyn, 'w') as f:
            f.write(self.dyn_script)
        sp.call("msander -O -i calc/{} -o calc/traj.out -p calc/prmtop -c calc/amber.{}.rst -x calc/amber.{}.nc".format(self.name_dyn,self.name,self.name),shell=True)
        # Generate All geometry
        int_script = "bonds\nangles\ndihedrals\n"
        with open('./calc/'+self.name_int, 'w') as f:
            f.write(int_script)
        sp.call("cpptraj -p calc/prmtop < calc/{} > calc/{}".format(self.name_int,self.name_geo),shell=True)
        self.extract(log)
        
        return
    def run(self,check_tokens=False):
        logger.log(5, 'RUNNING: {}'.format(self.filename))
        self._index_output_log = []
        com_opts = self.get_com_opts()
        current_directory = os.getcwd()
        os.chdir(self.directory)
        log = open(self.name_log,'w')
        sp.call("mkdir calc",shell=True, stderr=log, stdin=log, stdout=log)
        if com_opts['opt']:
            logger.log(logging.DEBUG, '  MINIMIZE & ANALYZE: {}'.format(self.filename))
            # Run leap
            sp.call("tleap -f {}".format(self.filename),shell=True, stderr=log, stdin=log, stdout=log) # parm7 rst7 files made
            # Run Min
            self.min_script = self.min_script.replace("aa","700")
            self.min_script = self.min_script.replace("bb","5")
            with open("./calc/"+self.name_min, 'w') as f:
                f.write(self.min_script)
            sp.call("msander -O -i calc/{} -o calc/{} -p calc/prmtop -c calc/inpcrd -r calc/amber.{}.rst".format(self.name_min,self.name_ene,self.name),shell=True, stderr=log, stdin=log, stdout=log)
        elif com_opts['sp']:
            logger.log(logging.DEBUG, '  CALCULATE: {}'.format(self.filename))
            # Run leap
            sp.call("tleap -f {}".format(self.filename),shell=True, stderr=log, stdin=log, stdout=log) # parm7 rst7 files made
            # Run Min
            self.min_script = self.min_script.replace("aa","0")
            self.min_script = self.min_script.replace("bb","0")
            with open("./calc/"+self.name_min, 'w') as f:
                f.write(self.min_script)
            sp.call("msander -O -i calc/{} -o calc/{} -p calc/prmtop -c calc/inpcrd -r calc/amber.{}.rst".format(self.name_min,self.name_ene,self.name),shell=True, stderr=log, stdin=log, stdout=log)
        # check if energy calculation failed
        restart = 1
        while(restart==1):
            with open("./calc/"+self.name_ene,'r') as f:
                fline = f.readlines()
                for line in fline:
                    if "restarting should resolve the error" in line:
                        sp.call("msander -O -i calc/{} -o calc/{} -p calc/prmtop -c calc/amber.{}.rst -r calc/amber.{}.rst".format(self.name_min,self.name_ene,self.name,self.name),shell=True, stderr=log, stdin=log, stdout=log)
                        restart = 1
                    else:
                        restart = 0

        if com_opts['geo']:
            self.geometry(log)
        if com_opts['freq']:
            self.hessian(log)
            # if geo file is already present 
            # may not have geo file if hessian is only ran
            if os.path.isfile('./calc/'+self.name_geo):
                self.geo_extract()
        os.chdir(current_directory)


#endregion AMBER I/O

def fetch_reference_data(parsed_ref_args) -> List[Datum]:
    GaussLog(path)
    #TODO unimplemented
    data:List[Datum] = []
    return data


#endregion File I/O
