#!/usr/bin/env python3
"""Estimate bond and angle parameters with the Seminario/FUERZA projection.

Projection always uses an unweighted Cartesian Hessian in Hartree/Bohr^2
and bond lengths in Bohr. Geometry files are in Angstrom; equilibrium bond
lengths and angles are written in Angstrom and degrees. Output conversion
is separate: Amber uses kcal/mol with a factor 1/2 for BOTH force constants;
MM3* uses the native STR/BND factors from its .fld header.

Amber input: .frcmod plus MOL2. MM3* input: .fld plus a MacroModel .mmo
listing containing coordinates and FF Line assignments (DEBG 56). Gaussian
frequency archives provide the reference geometry and raw Cartesian Hessian.
An optional --hessian .npy/text matrix can replace the archive matrix;
--reverse declares that supplied matrix to be mass weighted (masses in amu).

--invert VALUE temporarily mass weights the Hessian, replaces the TS
eigenvalue with VALUE in kJ/(mol*A^2*amu), as RDAT -gh -i does, and removes
the mass weighting before projection. This is independent of --reverse.

Adapted in part from https://github.com/SGenheden/Seminario.
"""

from __future__ import division, print_function, absolute_import
import argparse
import copy
import glob
import os
import shlex

import numpy as np

import math_util

import logging
import logging.config
import constants as co
import utilities
from utilities import GaussLog, Mol2
from data_structs import Atom, AmberFF, MM3Param

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# region calculation_methods

# TODO perhaps move some of these to linear_algebra.py,
# can do when I reformat the docstrings to Google style after I've tested the method.
# Preferably, have it done before publication though.


def sub_hessian(hessian, atom1: Atom, atom2: Atom, ang_to_bohr=False) -> tuple:
    """Subsample the Hessian matrix by pulling out the terms relevant to the
    bond that is formed between atom1 and atom2 as well as calculating the
    vector from atom1 to atom2.

    Args:
        hessian (np.ndarray): Hessian matrix
        atom1 (Atom): first atom
        atom2 (Atom): second atom
        ang_to_bohr (bool, optional): whether to convert the hessian terms from
         Angstrom length units to Bohr radii units. Defaults to False.

    Returns:
        tuple: tuple of (vector from atom1 to atom2, eigenvalues of the submatrix,
         eigenvector of the submatrix)
    """
    vec12 = atom1.coords - atom2.coords
    if ang_to_bohr:
        vec12 = vec12 / co.BOHR_TO_ANG
    norm = np.linalg.norm(vec12)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Bond projection needs distinct, finite atomic coordinates.")
    vec12 = vec12 / norm

    submat = -hessian[
        3 * (atom1.index - 1) : 3 * (atom1.index - 1) + 3,
        3 * (atom2.index - 1) : 3 * (atom2.index - 1) + 3,
    ]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec


def get_unit_vector(atom1: Atom, atom2: Atom, ang_to_bohr=False) -> np.ndarray:
    """Get the unit vector between atom1 and atom2

    Args:
        atom1 (Atom): first atom
        atom2 (Atom): second atom
        ang_to_bohr (bool, optional): whether to convert the distance from Angstroms to Bohr radii. Defaults to False.

    Returns:
        numpy.ndarray: unit vector between atom1 and atom2
    """
    vec12 = atom1.coords - atom2.coords
    vec21 = atom2.coords - atom1.coords
    unit_vec = np.hstack((vec12, vec21))
    if ang_to_bohr:
        unit_vec = unit_vec / co.BOHR_TO_ANG
    unit_vec = unit_vec / np.linalg.norm(unit_vec)
    return unit_vec


def get_subhessian(hessian, atom1: Atom, atom2: Atom) -> np.ndarray:
    """Subsample the Hessian matrix by pulling out the terms relevant to the
    bond that is formed between atom1 and atom2 as well as calculating the
    vector from atom1 to atom2.

    Args:
        hessian (np.ndarray): Hessian matrix
        atom1 (Atom): first atom
        atom2 (Atom): second atom

    Returns:
        np.ndarray: submatrix of the Hessian relevant to atom1-atom2 interactions.
    """
    submat_11 = (
        0
        * hessian[
            3 * atom1.index - 3 : 3 * atom1.index, 3 * atom1.index - 3 : 3 * atom1.index
        ]
    )
    submat_12 = -hessian[
        3 * atom1.index - 3 : 3 * atom1.index, 3 * atom2.index - 3 : 3 * atom2.index
    ]
    submat_21 = -hessian[
        3 * atom2.index - 3 : 3 * atom2.index, 3 * atom1.index - 3 : 3 * atom1.index
    ]
    submat_22 = (
        0
        * hessian[
            3 * atom2.index - 3 : 3 * atom2.index, 3 * atom2.index - 3 : 3 * atom2.index
        ]
    )
    submat_1 = np.hstack((submat_11, submat_12))
    submat_2 = np.hstack((submat_21, submat_22))
    submat = np.vstack((submat_1, submat_2))

    return submat


def seminario_sum(vec, eigval, eigvec) -> float:
    """Average the projections of the Hessian eigenvector on a specific interaction unit vector
    according to FUERZA

    Args:
        vec (np.ndarray): unit vector (atomic interaction)
        eigval (np.ndarray): eigenvalues of Hessian submatrix
        eigvec (np.ndarray): eigenvectors of Hessian submatrix

    Returns:
        float: the averaged projection to serve as a force constant estimate
    """
    ssum = 0.0
    for i in range(3):
        ssum += eigval[i] * np.abs(np.dot(eigvec[:, i], vec))
    return ssum


def seminario_bond(atoms: list, hessian, scaling=0.963, ang_to_bohr=True) -> float:
    """Estimate the bond force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix. Will average over atom1->atom2 and
    atom2->atom1 force constants.

    Args:
        atoms (list[Atom]): atoms involved in the bond for which to estimate a force constant
        hessian (np.ndarray): Unweighted Cartesian Hessian in Hartree/Bohr^2.
        scaling (float, optional): Curvature/energy multiplier. Defaults to 0.963.
        ang_to_bohr (bool, optional): Convert Angstrom coordinates to Bohr.

    Note:
        The default is the QFUERZA paper's energy scaling factor. A frequency
        scaling factor would need to be squared before multiplying curvature.

    Returns:
        float: estimated bond force constant in AU
    """

    vec12, eigval12, eigvec12 = sub_hessian(
        hessian, atoms[0], atoms[1], ang_to_bohr=ang_to_bohr
    )
    f12 = seminario_sum(vec12, eigval12, eigvec12)

    vec21, eigval21, eigvec21 = sub_hessian(
        hessian, atoms[1], atoms[0], ang_to_bohr=ang_to_bohr
    )

    f21 = seminario_sum(vec21, eigval21, eigvec21)

    if f12 <= 0 or f21 <= 0:
        logger.log(
            logging.DEBUG,
            "Nonpositive bond projections for atoms {}, {} (Hartree/Bohr^2): {} {}".format(
                atoms[0].index, atoms[1].index, f12, f21
            ),
        )

    if np.iscomplexobj(f12):
        logger.log(
            logging.DEBUG,
            "Complex number in estimate for bond f12 (" + str(f12) + "): " + str(atoms),
        )
        if not np.iscomplex(f12):
            f12 = np.real(f12)
        else:
            logger.log(
                logging.WARN,
                "WARNING: Non-zero imaginary component of bond force constant estimate for angle ("
                + str(f12)
                + "): "
                + str(atoms)
                + ".",
            )

    if np.iscomplexobj(f21):
        logger.log(
            logging.DEBUG,
            "Complex number in estimate for bond f21 (" + str(f21) + "): " + str(atoms),
        )
        if not np.iscomplex(f21):
            f21 = np.real(f21)
        else:
            logger.log(
                logging.WARN,
                "WARNING: Non-zero imaginary component of bond force constant estimate for angle ("
                + str(f21)
                + "): "
                + str(atoms)
                + ".",
            )

    f = 0.5 * (f12 + f21)
    if f <= 0:
        logger.log(
            logging.WARNING,
            "Nonpositive bond curvature for atoms {}, {} (raw, scaled Hartree/Bohr^2): {} {}".format(
                atoms[0].index, atoms[1].index, f, f * scaling
            ),
        )

    return scaling * f


def seminario_angle(
    atoms: list, hessian, scaling=0.963, convert=False, ang_to_bohr=True
) -> float:
    """Estimate the angle force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix.

    Args:
        atoms (list[Atom]): list of Atom objects for which to estimate the angle force constant
        hessian (numpy.ndarray): Unweighted Cartesian Hessian in Hartree/Bohr^2.
        scaling (float, optional): Curvature/energy multiplier. Defaults to 0.963.
        convert (bool, optional): whether to convert force constant from AU to kJ/mol. Defaults to False.
        ang_to_bohr (bool, optional): Convert Angstrom bond lengths to Bohr.

    Note:
        The default is the QFUERZA paper's energy scaling factor. A frequency
        scaling factor would need to be squared before multiplying curvature.

    Returns:
        float: angular curvature in Hartree/rad^2 (kJ/mol/rad^2 if convert=True).
    """

    assert len(atoms) == 3
    vec12, eigval12, eigvec12 = sub_hessian(hessian, atoms[0], atoms[1])
    vec32, eigval32, eigvec32 = sub_hessian(hessian, atoms[2], atoms[1])

    un = np.cross(vec32, vec12)
    norm = np.linalg.norm(un)
    if not np.isfinite(norm) or norm < 1e-8:
        logger.warning("Cannot project a linear or degenerate angle: %s", atoms)
        return np.nan
    un = un / norm
    upa = np.cross(un, vec12)
    upc = np.cross(vec32, un)

    sum1 = seminario_sum(upa, eigval12, eigvec12)
    sum2 = seminario_sum(upc, eigval32, eigvec32)

    len12 = utilities.measure_bond(atoms[0].coords, atoms[1].coords)
    if ang_to_bohr:
        len12 = len12 / co.BOHR_TO_ANG
    len32 = utilities.measure_bond(atoms[2].coords, atoms[1].coords)
    if ang_to_bohr:
        len32 = len32 / co.BOHR_TO_ANG

    if sum1 == 0 or sum2 == 0 or len12 == 0 or len32 == 0:
        return np.nan
    f12 = 1.0 / (sum1 * len12 * len12)
    f32 = 1.0 / (sum2 * len32 * len32)

    if f12 <= 0 or f32 <= 0:
        logger.log(
            logging.DEBUG,
            "Nonpositive angular compliance for atoms {}, {}, {} (1/Hartree): {}, {}".format(
                atoms[0].index, atoms[1].index, atoms[2].index, f12, f32
            ),
        )

    denominator = f12 + f32
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    f = 1.0 / denominator

    if f < 0:
        logger.log(
            logging.WARNING,
            "Nonpositive angle curvature for atoms {}, {}, {} (raw, scaled Hartree/rad^2): {} {}".format(
                atoms[0].index, atoms[1].index, atoms[2].index, f, f * scaling
            ),
        )

    if np.iscomplexobj(f):
        logger.log(
            logging.DEBUG,
            "Complex number in estimate for angle(" + str(f) + "): " + str(atoms),
        )
        if not np.iscomplex(f):
            f = np.real(f)
        else:
            logger.log(
                logging.WARN,
                "WARNING: Non-zero imaginary component of angle force constant estimate for angle ("
                + str(f)
                + "): "
                + str(atoms),
            )

    if convert:
        return scaling * co.HARTREE_TO_KJMOL * f
    else:
        return scaling * f


# endregion calculation_methods

# region Arguments


def _positive_number(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return value


def return_params_parser(add_help=True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-i", "--ff-in", required=True,
                        help="Amber .frcmod or MM3* .fld with OPT parameters.")
    parser.add_argument("-o", "--ff-out", required=True,
                        help="Output force-field path.")
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("-m", "--mol", nargs="+",
                          help="Amber MOL2 file(s), in the Gaussian archive's atom order and frame.")
    geometry.add_argument("--mmo", nargs="+",
                          help="MM3* MacroModel listing(s), with FF Line assignments and DEBG 56 coordinates.")
    parser.add_argument("-gl", "--log", nargs="+", required=True,
                        help="Gaussian frequency log(s), one per structure. Archive Hessians are unweighted AU.")
    parser.add_argument("--hessian", nargs="+",
                        help="Optional full 3N x 3N matrix file(s), .npy or whitespace text, replacing the archive Hessians.")
    parser.add_argument("--hessian-units", choices=(co.GAUSSIAN, co.KJMOLA),
                        default=co.GAUSSIAN,
                        help="Matrix-file units: AU=Hartree/Bohr^2 (default), KJMOLA=kJ/(mol*A^2); per amu with --reverse.")
    parser.add_argument("--reverse", action="store_true",
                        help="The --hessian matrices are mass weighted using masses in amu; unweight the full matrix before projection.")
    parser.add_argument("--invert", type=_positive_number, metavar="EIGENVALUE",
                        help="Replace the mass-weighted TS eigenvalue with this value in kJ/(mol*A^2*amu), as RDAT -gh -i.")
    parser.add_argument("--scale", type=_positive_number, default=0.963,
                        help="Curvature/energy multiplier (default 0.963); a frequency multiplier must be squared first.")
    parser.add_argument("--prep", action="store_true",
                        help="Zero selected torsions and, for MM3*, bond dipoles. MOL2 partial charges are retained.")
    parser.add_argument("--skip-dummy", action="store_true",
                        help="Skip parameters involving dummy atoms.")
    parser.add_argument("--raw-fuerza", action="store_true",
                        help="Legacy alias: this implementation uses the Seminario/FUERZA projection without empirical H-angle substitution.")
    parser.add_argument("--individualize", action="store_true",
                        help="Write a separate force field for every structure.")
    return parser


# region FC Derivers & Averagers

# region AMBER

def _matches_param(param, dof, type_dict):
    if isinstance(param, MM3Param):
        return dof.ff_row == param.ff_row
    return utilities.is_same_type_DOF(param.atom_types, type_dict[dof])


def _has_dummy_match(param, structures):
    for struct in structures:
        types = struct.get_DOF_atom_types_dict()
        for dof in struct.bonds + struct.angles:
            if _matches_param(param, dof, types) and any(
                    atom.is_dummy for atom in struct.get_atoms_in_DOF(dof)):
                return True
    return False


def _valid_curvature(value):
    value = np.real_if_close(value, tol=1000)
    return not np.iscomplexobj(value) and np.isfinite(value) and value > 0


def _derive_force_constant(param, structs, hessians, kind, ang_to_bohr, scaling):
    if len(structs) != len(hessians):
        raise ValueError("Each structure must have one Hessian.")
    values = []
    matched = 0
    method = seminario_bond if kind == "bonds" else seminario_angle
    for struct, hessian in zip(structs, hessians):
        types = struct.get_DOF_atom_types_dict()
        for dof in getattr(struct, kind):
            if _matches_param(param, dof, types):
                matched += 1
                value = method(struct.get_atoms_in_DOF(dof), hessian,
                               ang_to_bohr=ang_to_bohr, scaling=scaling)
                if _valid_curvature(value):
                    values.append(float(np.real(value)))
                else:
                    logger.warning("Invalid %s projection at FF row %s in %s: %s",
                                   kind, param.ff_row, struct.origin_name, value)
    if not values:
        logger.warning("FF row %s: %s; retaining its input value.", param.ff_row,
                       "no matching interactions" if not matched else "no valid projections")
        return None
    return float(np.mean(values))


def derive_bf_param(param, structs, hessians, ang_to_bohr=True, scaling=0.963):
    """Mean valid bond curvature in Hartree/Bohr^2, before FF conversion."""
    return _derive_force_constant(param, structs, hessians, "bonds", ang_to_bohr, scaling)


def derive_af_param(param, structs, hessians, ang_to_bohr=True, scaling=0.963):
    """Mean valid angular curvature in Hartree/rad^2, using lengths in Bohr."""
    return _derive_force_constant(param, structs, hessians, "angles", ang_to_bohr, scaling)


def _average_equilibrium(param, structs, kind):
    values = []
    for struct in structs:
        types = struct.get_DOF_atom_types_dict()
        for dof in getattr(struct, kind):
            if _matches_param(param, dof, types) and np.isfinite(dof.value):
                values.append(dof.value)
    return float(np.mean(values)) if values else param.value


def average_ae_param(param, structs):
    """Mean equilibrium angle in degrees, matched by type or assigned MM3 row."""
    return _average_equilibrium(param, structs, "angles")


def average_be_param(param, structs):
    """Mean equilibrium bond length in Angstrom, independent of Bohr projection."""
    return _average_equilibrium(param, structs, "bonds")


def prepare_hessian(hessian, atoms, hessian_units=co.GAUSSIAN,
                    reverse=False, invert=None):
    """Return an unweighted Cartesian Hessian in Hartree/Bohr^2.

    reverse=True declares a mass-weighted input (amu). Inversion temporarily
    uses mass-weighted AU/amu, while its supplied replacement has RDAT's
    kJ/(mol*A^2*amu) units. Neither operation changes output FF conventions.
    """
    hessian = np.array(hessian, dtype=float, copy=True)
    if hessian.shape != (3 * len(atoms), 3 * len(atoms)):
        raise ValueError("Hessian dimensions do not match the reference atoms.")
    if not np.isfinite(hessian).all() or not np.allclose(
            hessian, hessian.T, rtol=1e-8, atol=1e-10):
        raise ValueError("Hessian must be finite and symmetric.")
    if hessian_units == co.KJMOLA:
        hessian /= co.HESSIAN_CONVERSION
    elif hessian_units != co.GAUSSIAN:
        raise ValueError("Hessian units must be AU or KJMOLA.")
    if reverse or invert is not None:
        try:
            masses = np.array([co.MASSES[atom.element] for atom in atoms])
        except KeyError as error:
            raise ValueError("Unknown atomic mass: {}".format(error)) from error
        if not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("Mass weighting requires positive atomic masses in amu.")
        sqrt_mass = np.repeat(np.sqrt(masses), 3)
        mass_matrix = np.outer(sqrt_mass, sqrt_mass)
        if reverse:
            hessian *= mass_matrix
        if invert is not None:
            if not np.isfinite(invert) or invert <= 0:
                raise ValueError("Inversion eigenvalue must be finite and positive.")
            mw = hessian / mass_matrix
            values, vectors = np.linalg.eigh(mw)
            # About 5 cm^-1: ignore tiny numerical curvatures when checking
            # that inversion has a negative direction to act on.
            significant = values < -1e-6
            count = np.count_nonzero(significant)
            if not count:
                raise ValueError("TS inversion needs a significant negative eigenvalue.")
            if count > 1:
                logger.warning("Hessian has %s significant negative eigenvalues; replacing only the lowest and retaining the others.", count)
            values[0] = invert / co.HESSIAN_CONVERSION
            hessian = (vectors @ np.diag(values) @ vectors.T) * mass_matrix
    return hessian


def curvature_to_parameter(value, ptype, force_field):
    """Convert AU curvature to a native coefficient, after projection.

    The Bohr/Angstrom conversion is dimensional. Amber's separate 1/2
    follows E=K*dq^2, whereas the projected curvature is d2E/dq2. MM3*
    instead uses its own STR/BND header factors, with no extra Amber 1/2.
    """
    if ptype not in ("bf", "af"):
        raise ValueError("Only bond and angle curvatures can be converted.")
    dimensional = (co.HESSIAN_CONVERSION if ptype == "bf"
                   else co.HARTREE_TO_KJMOL)
    curvature_kj = value * dimensional
    if force_field.units == co.AMBERFF:
        return 0.5 * curvature_kj * co.HARTREE_TO_KCALMOL / co.HARTREE_TO_KJMOL
    if force_field.units == co.MM3FF:
        return curvature_kj / force_field.curvature_factors[ptype]
    raise ValueError("Unsupported force field.")


def _estimate_force_field(force_field, structures, hessians, zero_out=False,
                          hessian_units=co.GAUSSIAN, skip_dummy=False,
                          reverse=False, invert=None, scaling=0.963):
    if not structures or len(structures) != len(hessians):
        raise ValueError("Each structure must have one Hessian.")
    if not np.isfinite(scaling) or scaling <= 0:
        raise ValueError("Curvature scale must be finite and positive.")
    hessians = [prepare_hessian(h, s.atoms, hessian_units, reverse, invert)
                for s, h in zip(structures, hessians)]
    estimated_ff = copy.deepcopy(force_field)
    n_estimated = 0
    for param in estimated_ff.params:
        if skip_dummy and (any(t in ("D1", "Du", "X") for t in param.atom_types)
                           or _has_dummy_match(param, structures)):
            continue
        if param.ptype == "be":
            param.value = average_be_param(param, structures)
        elif param.ptype == "ae":
            param.value = average_ae_param(param, structures)
        elif zero_out and param.ptype in ("df", "q"):
            param.value = 0.0
        elif param.ptype in ("bf", "af"):
            derive = derive_bf_param if param.ptype == "bf" else derive_af_param
            value = derive(param, structures, hessians, ang_to_bohr=True,
                           scaling=scaling)
            if value is not None:
                param.value = curvature_to_parameter(value, param.ptype, force_field)
                n_estimated += 1
    if not n_estimated:
        raise ValueError("No valid matched force constants; check atom types, MMO FF rows and Hessians.")
    return estimated_ff


def amber_qfuerza(force_field, structures, hessians, zero_out=False,
                   hessian_units=co.GAUSSIAN, skip_dummy=False,
                   reverse=False, invert=None, scaling=0.963):
    return _estimate_force_field(force_field, structures, hessians, zero_out,
                                 hessian_units, skip_dummy, reverse, invert, scaling)


def mm3_qfuerza(force_field, structures, hessians, zero_out=False,
                 hessian_units=co.GAUSSIAN, skip_dummy=False,
                 reverse=False, invert=None, scaling=0.963):
    return _estimate_force_field(force_field, structures, hessians, zero_out,
                                 hessian_units, skip_dummy, reverse, invert, scaling)


# Kept for callers of the old API; both use the original projection.
amber_raw_fuerza = amber_qfuerza


def _expand_paths(paths):
    result = []
    for pattern in paths:
        found = sorted(glob.glob(pattern))
        if not found:
            raise ValueError("No files match: {}".format(pattern))
        result.extend(found)
    return result


def _use_reference_geometry(structure, reference):
    """Use exact QM coordinates after checking atom order and orientation."""
    source = np.array([a.coords for a in structure.atoms], dtype=float)
    target = np.array([a.coords for a in reference.atoms], dtype=float)
    if source.shape != target.shape or not np.isfinite(source).all():
        raise ValueError("Structure and Gaussian reference have different atoms/coordinates.")
    if not np.allclose(source - source.mean(axis=0), target - target.mean(axis=0),
                       atol=1e-3, rtol=0):
        raise ValueError("{} is not in the Gaussian archive's atom order and frame."
                         .format(structure.origin_name))
    for atom, qm_atom in zip(structure.atoms, reference.atoms):
        atom.coords = qm_atom.coords.copy()
        atom._element = qm_atom.element
    for bond in structure.bonds:
        a, b = structure.get_atoms_in_DOF(bond)
        bond.value = math_util.measure_bond(a.coords, b.coords)
    for angle in structure.angles:
        a, b, c = structure.get_atoms_in_DOF(angle)
        u, v = a.coords - b.coords, c.coords - b.coords
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        angle.value = np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))) if norm else np.nan


def main(args=None):
    if isinstance(args, str):
        args = shlex.split(args)
    parser = return_params_parser()
    args = parser.parse_args(args)
    try:
        if args.ff_in.endswith(".frcmod"):
            if not args.mol:
                raise ValueError("Amber .frcmod input requires --mol.")
            ff = utilities.AmberUtilities.read_frcmod(args.ff_in)
            if not ff.params:
                raise ValueError("No # Q2MM / OPT parameters in the frcmod.")
            structures = [s for path in _expand_paths(args.mol) for s in Mol2(path).structures]
        elif args.ff_in.endswith(".fld"):
            if not args.mmo:
                raise ValueError("MM3* .fld input requires --mmo for exact FF Line assignments.")
            ff = utilities.MM3Utilities.read_fld(args.ff_in)
            structures = [s for path in _expand_paths(args.mmo)
                          for s in utilities.MM3Utilities.read_mmo(path)]
            for struct in structures:
                for kind, dofs in (("1", struct.bonds), ("2", struct.angles)):
                    for dof in dofs:
                        if not (1 <= dof.ff_row <= len(ff.lines)) or ff.lines[dof.ff_row-1][1:2] != kind:
                            raise ValueError("MMO FF Line {} does not match this .fld file."
                                             .format(dof.ff_row))
        else:
            raise ValueError("Input force field must be .frcmod (Amber) or .fld (MM3*).")
        logs = [GaussLog(path, au_hessian=True) for path in _expand_paths(args.log)]
        if len(logs) != len(structures):
            raise ValueError("Supply one Gaussian log for each structure.")
        if args.reverse and not args.hessian:
            raise ValueError("Gaussian archive Hessians are unweighted; --reverse requires --hessian matrix input.")
        if not args.hessian and args.hessian_units != co.GAUSSIAN:
            raise ValueError("--hessian-units applies only to --hessian matrix files.")
        hessians = []
        for struct, log in zip(structures, logs):
            reference = log.structures[-1]
            _use_reference_geometry(struct, reference)
            hessians.append(reference.hess)
        if args.hessian:
            paths = _expand_paths(args.hessian)
            if len(paths) != len(structures):
                raise ValueError("Supply one --hessian matrix per structure.")
            hessians = [np.load(path, allow_pickle=False) if path.endswith(".npy")
                        else np.loadtxt(path) for path in paths]
        elif any(h is None for h in hessians):
            raise ValueError("Gaussian log does not contain an archive Hessian.")
        estimate = amber_qfuerza if isinstance(ff, AmberFF) else mm3_qfuerza
        if args.raw_fuerza:
            logger.info("Using original Seminario/FUERZA projections (no empirical H-angle substitution).")
        groups = [([s], [h]) for s, h in zip(structures, hessians)] if args.individualize else [(structures, hessians)]
        outputs = []
        for index, (structs, matrices) in enumerate(groups, 1):
            estimated = estimate(ff, structs, matrices, args.prep,
                                 hessian_units=args.hessian_units,
                                 skip_dummy=args.skip_dummy, reverse=args.reverse,
                                 invert=args.invert, scaling=args.scale)
            path = args.ff_out
            if args.individualize:
                directory, name = os.path.split(path)
                path = os.path.join(directory, "{}.{}".format(index, name))
            if isinstance(estimated, AmberFF):
                # Do not let the shared writer silently retain old values.
                if any(not np.isfinite(p.value) or abs(p.value) > 1999 for p in estimated.params):
                    raise ValueError("Estimated Amber parameter is nonfinite or exceeds the writer's 1999 limit.")
                utilities.AmberUtilities.write_frcmod(estimated, path, estimated.params)
            else:
                utilities.MM3Utilities.write_fld(estimated, path)
            outputs.append(path)
        return outputs
    except (ValueError, OSError, IndexError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
