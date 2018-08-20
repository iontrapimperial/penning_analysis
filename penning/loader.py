"""
Provides the `load()` and `load_many()` functions, which are accessible from the
package root.
"""

from . import SpectrumDataFile, independents, probabilities
from .spectrum import metadata_fields
import pathlib
import numpy as np

__all__ = ['load', 'load_many']

def load(file: str, override_shots:int=None) -> SpectrumDataFile:
    """
    Given a path to a data file, read the metadata and data into Python types
    and return the resulting class.  The number of shots per point can be
    overridden with the `override_shots` argument, which takes an integer.

    Arguments:
    file: str -- The file name to load from.
    override_shots: ?int --
        The number of shots per point to use.  This value takes precedence over
        the number found in the file.  It is a `ValueError` to try and override
        to a number of shots which doesn't divide cleanly into the total number
        of acquisitions in the file.

    Returns:
    SpectrumDataFile --
        The Python representation of the output data file.

    Raises:
    ValueError -- If the override number of shots is invalid for the file.
    """
    metadata = []
    with open(file, "r") as f:
        line = 0
        for description, _, parse, type_, _ in metadata_fields:
            line += 1
            file_description = f.readline().rstrip()
            if description != file_description:
                raise ValueError(
                    f"Unexpected field identifier on line {line} of '{file}'."
                    f"  Expected '{description}', but got '{file_description}'."
                )
            line += 1
            value = f.readline().rstrip()
            try:
                metadata.append(parse(value))
            except ValueError:
                raise ValueError(
                    f"Unable to parse value on line {line} of '{file}'."
                    f"  Value was '{value}', but expected type '{type_}'."
                )
        line += 1
        data_line = f.readline().rstrip()
        data_string = 'Data:'
        if data_line != data_string:
            raise ValueError(
                f"Expected to see the data identifier '{data_string}' on line"
                f" {line}, but instead saw '{data_line}'."
            )
        data = np.array([int(line) for line in f], dtype=np.int32)
        data = np.transpose(np.reshape(data, (data.shape[0] // 4, 4)))
        data = np.core.records.fromarrays(
                    data,
                    names='cool, cool_error, count, count_error',
                    formats='i4, i4, i4, i4')
    data_file = SpectrumDataFile(data, metadata, file)
    if override_shots is not None:
        total_shots = data_file.shots * data_file.points
        if total_shots % override_shots != 0:
            raise ValueError("Could not override the number of shots per point"\
                             + f" to be {override_shots} when the file has"\
                             + f" {total_shots} total acquisitions.")
        data_file.shots = override_shots
        data_file.points = total_shots // override_shots
    return data_file

def load_many(id, cool_threshold, count_threshold, override_shots:int=None,
              dir=".") -> ([SpectrumDataFile], np.array, np.array):
    """
    Load many readings files into one coherent set of independent variables and
    measures.  The files, independents and measures are returned as a 3-tuple in
    that order.

    Arguments --
        id: str --
            The identifier of the files you want to open.  This is typically a
            three digit number.  The loader will open all files in the given
            directory which match the glob '{id}*.txt'.
        cool_threshold: int --
            The minimum allowable count rate during the cooling cycle.
        count_threshold: int --
            The minimum count rate that a shot can have to be detected in the
            "light" state.
        override_shots: ?int --
            The actual number of shots to use when loading each file.  This
            overrides any value set in the file itself.
        dir: ?str --
            The directory to search for the readings files in.

    Returns --
    (files, independents, measures) --
        files: list of SpectrumDataFile --
            An unordered list of the data files that were found and loaded.
        independents: np.array of float (as angular Hz or s) --
            The independent variables sampled in all of the data files, ordered
            from smallest to highest.
        probabilities: 2D np.array of (probability: float, error: float) --
            An array of the probabilities and errors for the excitation of the
            ion(s).  The first index runs over the number of ions excited.  For
            more information, see the `probabilities()` function in the package
            root.

    Raises --
        ValueError --
            - If there are no files found matching the id.
            - If the number of `override_shots` doesn't divide cleanly into one
              of the data files.
    """
    dir = pathlib.Path(dir)
    files = [load(file, override_shots) for file in dir.glob(id + "*.txt")]
    if len(files) == 0:
        raise ValueError(f"Could not find any files in directory {str(dir)}"
                         + f" which match the id {id}.")
    xs = np.concatenate([independents(file) for file in files])
    ys = np.concatenate([probabilities(file, cool_threshold, count_threshold)\
                         for file in files], axis=1)
    order = xs.argsort()
    return files, xs[order], ys[:,order]
