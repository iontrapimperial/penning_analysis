"""
Functions for extracting the stored data from a `DataFile`.  The base function
is `independents()`, which automatically detects the type of the independent
parameter (and is available in the package namespace).  You can manually call
`frequencies()` and `times()` if this generation is not successful.
"""

from . import _helpers, DataFile
import numpy as np

__all__ = ['probabilities', 'independents', 'frequencies', 'times']

def probabilities(data_file: DataFile,
                  cool_threshold: int,
                  count_thresholds: int,
                  min_error: float=0.01) -> np.array:
    return np.transpose(np.array(
        [_helpers.point_probabilities(point, cool_threshold, count_thresholds,
                                      min_error)
         for point in _helpers.points(data_file.data, data_file.shots)]))
probabilities.__doc__ =\
    f"""
    Get the probabilities and errors of excitation of any number of ions for all
    the points in the data file.

    Arguments --
    data_file: DataFile -- An output data file loaded using `penning.load`.
    {_helpers.doc_thresholds.strip()}

    Returns --
    np.array(shape=(n_ions + 1, n_points),
             dtype=[("probability", "f8"), ("error", "f8")]) --
        An array of the probabilities and errors for each possible number of
        ions, for every point in the data file.  The array can be indexed using
        `[n_excited, n_point]`, with `n_excited` being the number of excited
        ions and `n_point` being the point number to look at.  At any stage of
        indexing, you can also de-structure the array by indexing (using a
        separate `[]`) on the field name you're interested in.  If you don't use
        a field index, at the end you will get out a `(probability, error)`
        tuple.

        In the following examples, we consider a single ion, where 3 points of
        data were taken.

        This looks at the probabilities that 0 ions are in the excited state:
            out[0] = array([
                (0.25773916, 0.04440978),
                (0.34375,    0.04847529),
                (0.28,       0.04489989)])
            #    ^           ^ error in the probability
            #    | probability that 0 ions are in the excited state
        The next example looks at all the data available for a particular point:
            out[:, 0] = array([
                (0.25773196, 0.0444078),   # prob, error of 0 ions excited
                (0.75226804, 0.0444078)])  # prob, error of 1 ion excited
            #    ^           ^ error in probability
            #    | probability that n ions are excited
        We can also extract the probabilities for all points, for all numbers of
        ions:
            out['probability'] = array([
                [0.25773916, 0.34375, 0.28],  # probabilities that 0 are excited
                [0.74226804, 0.65625, 0.72]]) # probabilities that 1 is excited
            #    ^           ^        ^ point 2
            #    |           | point 1
            #    | point 0
        We can use the field identifier indexing at any point.  This gives the
        errors in the probabilities of 1 ion being excited, for each of the
        points:
            out[1]['error'] = array([0.04440978, 0.04847529, 0.04489989])
    """

def frequencies(data_file: DataFile) -> np.array:
    """
    Get the frequencies used for each point in the scan.  It doesn't make sense
    to call this function if the independent parameter was wait time.  Consider
    using `independents()` instead for automatic detection.
    """
    start_offset = data_file.aom_start - data_file.carrier
    return start_offset + data_file.step_size * np.arange(data_file.points)

def times(data_file: DataFile) -> np.array:
    """
    Get the times used for each point in the scan.  It doesn't make sense to
    call this function if the independent parameter was frequency.  Consider
    using `independents()` instead for automatic detection.
    """
    return data_file.start_time\
           + data_file.step_size * np.arange(data_file.points)

_independents = {
    "Continuous": frequencies,
    "Windowed":   frequencies,
    "Fixed":      times,
}

def independents(data_file: DataFile) -> np.array:
    """
    Return an array of the independent parameters for each point in this
    `DataFile`.  If the spectrum is a frequency scan, this will be an array of
    frequencies in Hz.  If it scans the time (e.g. in a Rabi), it will be an
    array of times in s.

    The type of parameter is detected automatically.  You can manually override
    the selection by calling the relevant function in the `penning.spectrum`
    module.
    """
    return _independents[data_file.type](data_file)
