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

# endregion Hessian-specific
