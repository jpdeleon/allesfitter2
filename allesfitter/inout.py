#!/usr/bin/env python3
"""
Input/Output utilities for allesfitter.

This module provides functions for reading and writing various file formats
including CSV, JSON, and pickle files. It is used throughout allesfitter
for data loading and results persistence.

Functions:
    write_csv: Write multiple arrays to a CSV file.
    read_csv: Read a CSV file and unpack columns.
    write_json: Write a dictionary to a JSON file.
    read_json: Read a JSON file into a dictionary.
    write_pickle: Write an object to a pickle file.
    read_pickle: Read a pickle file into an object.
"""


#::: modules
import json
from typing import Any, Dict, Tuple

import numpy as np

#::: plotting settings
import seaborn as sns

sns.set(
    context="paper",
    style="ticks",
    palette="deep",
    font="sans-serif",
    font_scale=1.5,
    color_codes=True,
)
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})
sns.set_context(rc={"lines.markeredgewidth": 1})


def write_csv(fname: str, *arrays: np.ndarray, **kwargs: Any) -> None:
    """Write multiple arrays to a CSV file.

    Parameters
    ----------
    fname : str
        Name of the output file (including path).
    *arrays : np.ndarray
        One or more arrays to write as columns.
    **kwargs : Any
        Additional keyword arguments passed to np.savetxt (e.g., fmt).

    Returns
    -------
    None

    Examples
    --------
    >>> write_csv('output.csv', time, flux, flux_err, fmt=['%.18e','%.12e','%.12e'])
    """
    X = np.column_stack(arrays)
    np.savetxt(fname, X, delimiter=",", **kwargs)


def read_csv(fname: str, skip_header: int = 0) -> Tuple[np.ndarray, ...]:
    """Read a CSV file and unpack the columns.

    Parameters
    ----------
    fname : str
        Name of the input CSV file.
    skip_header : int, optional
        Number of header lines to skip (default: 0).

    Returns
    -------
    tuple of np.ndarray
        The columns of the CSV file as separate arrays.

    Examples
    --------
    >>> time, flux, flux_err = read_csv('data.csv')
    """
    return np.genfromtxt(
        fname,
        delimiter=",",
        comments="#",
        encoding="utf-8",
        dtype=float,
        unpack=True,
        skip_header=skip_header,
    )


def write_json(fname: str, dic: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file.

    Parameters
    ----------
    fname : str
        Name of the output JSON file.
    dic : dict
        Dictionary to serialize to JSON.

    Returns
    -------
    None
    """
    with open(fname, "w") as fp:
        json.dump(dic, fp, indent=4)


def read_json(fname: str) -> Dict[str, Any]:
    """Read a JSON file into a dictionary.

    Parameters
    ----------
    fname : str
        Name of the input JSON file.

    Returns
    -------
    dict
        The JSON contents as a dictionary.
    """
    with open(fname) as fp:
        return json.load(fp)
