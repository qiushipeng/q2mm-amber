"""Contains methods which perform linear algebraic operations.

"""
from __future__ import division, print_function, absolute_import
import copy
import logging
from logging import config
import sys

import numpy as np
import constants as co
from typing import Tuple

config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# Print out full matrices rather than having Numpy truncate them.
# np.nan seems to no longer be supported for untruncated printing
# of arrays. The suggestion is to use sys.maxsize but I haven't checked
# that this works for python2 so leaving the commented code for now.
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)

# region Generalized


def measure_bond(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Returns bond length between 2 sets of coordinates.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]

    Returns:
        float: measured bond length
    """
    vector = coords2 - coords1
    return np.sqrt(
        vector.dot(vector)
    )  # Used over np.linalg.norm due to speed advantage


def measure_angle(
    coords1: np.ndarray, coords2: np.ndarray, coords3: np.ndarray
) -> float:
    """Returns angle between 3 sets of coordinates in degrees.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]
        coords3 (np.ndarray): atom3 coordinates [x, y, z]

    Returns:
        float: Angle between coords1, coords2, coords3 in degrees
    """
    vector21 = coords1 - coords2
    vector23 = coords3 - coords2
    cos_angle = np.dot(vector21, vector23) / (
        np.sqrt(vector21.dot(vector21)) * np.sqrt(vector23.dot(vector23))
    )
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


# endregion Generalized

# region Hessian-specific (Hermitian)

def decompose(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decomposes matrix into its eigenvalues and eigenvectors.

    Args:
        matrix (np.ndarray): Matrix to decompose, matrix must be square.

    Returns:
        (np.ndarray, np.ndarray): (eigenvalues, eigenvectors) where eigenvalues
         is of shape (1,n) and eigenvectors is of shape (n,n) with n rows of
         eigenvectors of length n.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def replace_neg_eigenvalue(
    eigenvalues: np.ndarray, replace_with=5000.0, zer_out_neg=False, units=co.KJMOLA
) -> np.ndarray:
    """Replaces the most negative eigenvalue with a strong positive value to invert the curvature of the Potential Energy Surface.

    Args:
        eigenvalues (np.ndarray): Eigenvalues
        replace_with (float, optional): Value which should replace the most negative eigenvalue. Defaults to 1.0.
        zer_out_neg (bool, optional): If True, will zero out remaining negative eigenvalues. Defaults to False.
        units (_type_, optional): Units in which replaced eigenvalue should be returned. Defaults to co.KJMOLA.

    Returns:
        np.ndarray: Eigenvalues with most negative eigenvalue replaced and, if requested, remaining negative values zeroed out.
    """    
    neg_indices = np.argwhere([eval < 0 for eval in eigenvalues])

    if len(neg_indices) > 1:
        logger.log(logging.WARN, "more than one neg. eigenvalue: " + str([eigenvalues[index] for index in neg_indices]))
        index_to_replace = np.argmin(eigenvalues)
    else:
        index_to_replace = neg_indices[0][0]
    replaced_eigenvalues = copy.deepcopy(eigenvalues)

    if zer_out_neg:
        for neg_index in neg_indices:
            replaced_eigenvalues[neg_index[0]] = 0.00
    logger.log(logging.INFO,"max eval: "+str(max(replaced_eigenvalues)))
    replaced_eigenvalues[
        index_to_replace
    ] = replace_with * co.HESSIAN_CONVERSION  if units == co.KJMOLA else replace_with # TODO: MF determine if we stick to this method, what it depends on, etc
    logger.log(logging.INFO, "most negative eigenvalue replaced with "+str(replaced_eigenvalues[index_to_replace]))
    logger.log(logging.INFO, str([replaced_eigenvalues[index] for index in neg_indices]))

    return replaced_eigenvalues


def reform_hessian(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Forms the Hessian matrix by multiplying the eigenvalues and eigenvectors

    Args:
        eigenvalues (np.ndarray[float]): eigenvalues
        eigenvectors (np.ndarray[float]): eigenvectors

    Returns:
        np.ndarray: Hessian matrix
    """
    reformed_hessian = eigenvectors.dot(np.diag(eigenvalues).dot(eigenvectors.T))
    return reformed_hessian


def invert_ts_curvature(hessian_matrix: np.ndarray, replace_with=5000) -> np.ndarray:
    """Inverts the curvature of the Hessian matrix

    Args:
        hessian_matrix (np.ndarray): hessian matrix whose curvature to invert, presumed in KJMOLA

    Returns:
        np.ndarray: inverted hessian matrix
    """
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    inv_curv_hessian = reform_hessian(
        replace_neg_eigenvalue(eigenvalues, zer_out_neg=True, replace_with=replace_with), eigenvectors
    )

    #check_evals = np.diag()

    if not inv_curv_hessian.all() >= 0.0:
        logger.log(logging.WARN, "Inverted Hessian has negative values...")
        logger.log(logging.WARN, str(sum(inv_curv_hessian > 0))+" negative values...")

    return inv_curv_hessian


def replace_lowest_eigenvalue(eigenvalues: np.ndarray, value: float, label: str = "") -> np.ndarray:
    """Returns a copy of `eigenvalues` with the most negative one, the
    transition-state reaction coordinate, replaced by `value`. This is the -i
    flag of calculate: the value is used as given, in the eigenvalues' own
    units, and any other negative eigenvalues are left alone (unlike
    replace_neg_eigenvalue, which zeroes them and converts units).

    Use argmin(w), NOT argmin(|w|): a raw Hessian still carries ~0
    translation/rotation modes, so smallest-magnitude would grab a rigid-body
    mode and leave the real negative mode untouched.

    Args:
        eigenvalues (np.ndarray): eigenvalues, in any order
        value (float): eigenvalue to put in place of the most negative one
        label (str, optional): name of the source, for the warning

    Returns:
        np.ndarray: the eigenvalues with the replacement made
    """
    eigenvalues = np.array(eigenvalues, dtype=float)
    index = int(np.argmin(eigenvalues))
    if eigenvalues[index] >= 0.0:
        logger.warning(
            "invert requested but no negative eigenvalue in {} "
            "(min eig {:.4g}); not a transition state?".format(label, eigenvalues[index]))
    eigenvalues[index] = float(value)
    return eigenvalues


def invert_lowest_eigenvalue(hessian_matrix: np.ndarray, value: float, label: str = "") -> np.ndarray:
    """Replaces the most negative eigenvalue of the Hessian with `value`
    (see replace_lowest_eigenvalue) and reforms the matrix.

    Args:
        hessian_matrix (np.ndarray): symmetric Hessian matrix
        value (float): eigenvalue to put in place of the most negative one
        label (str, optional): name of the Hessian's source, for the warning

    Returns:
        np.ndarray: the reformed Hessian
    """
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    return reform_hessian(replace_lowest_eigenvalue(eigenvalues, value, label), eigenvectors)


def project_hessian(hessian: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """The Hessian expressed in the basis of `eigenvectors`: V H V^T.

    With the normalized, mass-weighted eigenvectors of the reference (QM)
    Hessian as rows of V (one per normal mode, n_modes x 3N), the result is
    the n_modes x n_modes "eigenmatrix" of eigenmode fitting: its diagonal is
    the curvature of `hessian` along each QM mode and its off-diagonals the
    coupling between modes. For the QM Hessian itself it is diag(eigenvalues);
    a force-field Hessian that reproduced the QM one would give the same.

    Args:
        hessian (np.ndarray): 3N x 3N mass-weighted Hessian
        eigenvectors (np.ndarray): n_modes x 3N, one eigenvector per row

    Returns:
        np.ndarray: n_modes x n_modes matrix

    Raises:
        ValueError: when the eigenvectors are not 3N long
    """
    eigenvectors = np.asarray(eigenvectors, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    if eigenvectors.ndim != 2 or hessian.ndim != 2 or eigenvectors.shape[1] != hessian.shape[0]:
        raise ValueError("eigenvectors of shape {} cannot project a Hessian of shape {}".format(
            eigenvectors.shape, hessian.shape))
    return eigenvectors.dot(hessian).dot(eigenvectors.T)

# endregion Hessian-specific
