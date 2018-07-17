"""
Helper functions for the spectrum analysis functions in the rest of this
package.
"""

import numpy as np

def points(data: np.array, shots: int) -> list:
    """
    Divide up the data in a `DataFile` into a list of points.  Each point is a
    `numpy.array` with `shots` elements in it.  Each shot is a structure tuple
    with the same fields as the original data.

    Arguments --
    data: numpy.array_like(DataFile.data) --
        The data to be split into points.  This can directly be the data in the
        `DataFile.data` field, or it can be data that has been de-interleaved
        into separate spectra (if that was necessary).
    shots: int -- The number of shots per point.  See `DataFile.shots`.

    Returns --
    list of numpy.array_like(DataFile.data) --
        A list of separated `numpy` arrays, where the list index runs over the
        points.
    """
    return np.split(data, data.shape[0] // shots)

def spectra(data: np.array, spectra: int) -> list:
    """
    De-interleave spectra from output data.  Returns a list of the separated
    spectra.

    Arguments --
    data: numpy.array_like(DataFile.data) --
        The data to be split into spectra.  This can directly be the data in the
        `DataFile.data` field, or it can be data that has been split already in
        some other manner (if necessary).
    spectra: int --
        The number of spectra which are interleaved.  See `DataFile.spectra`.

    Returns --
    list of numpy.array_like(DataFile.data) --
        A list of the data separated out into individual spectra.  The list
        index runs over the spectra.
    """
    return [data[spectrum::spectra] for spectrum in range(spectra)]

doc_thresholds =\
    """
    cool_threshold: int --
        The cooling threshold of validity for a shot.  If the `cool` field is
        less than this threshold, that shot will be discarded.

    count_thresholds: int | array_like of int --
        The threshold number of counts to distinguish between the dark and the
        light states of the ion(s).  There should be as many elements of
        `count_thresholds` as there are ions to distinguish between.  If the
        `count` value is less than or equal to the `n`th threshold value, then
        we assume that there are `n` ions in the ground state (where `n` counts
        from 0).

        For example, if we have
            count_thresholds = 4
        this implies that there is a single ion.  Then the following counts will
        imply these number of ions in ground and excited:
            count   ground  excited
            0       0       1
            4       0       1
            5       1       0
            ...
        If instead we have
            count_thresholds = [4, 8]
        this implies that there are two ions.  The following counts will imples
        these number of ions in ground and excited:
            count   ground  excited
            0       0       2
            4       0       2
            6       1       1
            8       1       1
            10      2       0
            ...

    min_error: float -- The minimum error to return for any probability measure.
    """

def point_probabilities(point, cool_threshold, count_thresholds, min_error):
    if not hasattr(count_thresholds, '__iter__'):
        count_thresholds = [count_thresholds]
    n_ions = len(count_thresholds)
    def n_ground_ions(count):
        """How many ions are in the ground state?  In other words, every time we
        cross a count threshold, we assume that one more ion is not excited, so
        here we just count up how many thresholds we cross until we match."""
        for i, threshold in enumerate(count_thresholds):
            if count <= threshold:
                return i
        else:
            return len(count_thresholds)
    # boolean mask to pull out valid shots
    cooling_mask = point['cool'] >= cool_threshold
    error_mask = np.logical_not(np.logical_or(point['cool_error'],
                                              point['counts_error']))
    shots = point[np.logical_and(cooling_mask, error_mask)]
    probabilities = np.zeros(n_ions + 1, dtype=np.float64)
    for count in shots['counts']:
        probabilities[n_ions - n_ground_ions(count)] += 1
    nshots = shots.shape[0]
    probabilities = probabilities / nshots
    errors = np.max([np.full_like(probabilities, min_error),
                     np.sqrt(probabilities * (1 - probabilities) / nshots)],
                    axis=0)
    return np.array(list(zip(probabilities, errors)),
                    dtype=[("probability", "f8"), ("error", "f8")])
point_probabilities.__doc__ =\
    f"""
    Get the probabilities and errors of excitation for any number of ions, from
    a single point of data.

    Arguments --
    point: np.array(dtype=[("cool", "i4"), ("cool_error", "i4"),
                           ("counts", "i4"), ("counts_error", "i4")]) --
        A point of data.  Each row in the array corresponds to one shot from the
        point, and the structured fields are the same form as is read in when
        the data file is loaded.

    {doc_thresholds.strip()}

    Returns --
    np.array(shape=(n_ions + 1,),
             dtype=[("probability", "f8"), ("error", "f8")]) --
        An array of the probabilities and errors of excitations.  The `n`th
        element of the output array is the probability and error that `n` ions
        were excited.  The sum of the probabilities in the array will always be
        equal to 1.
    """
