"""
Contains the class `DataFile`, which is a Python representation of the data and
the metadata.  The loading of these is handled by `load()` and `load_many()` in
the package root.

Also contains functions for extracting the stored data from a `DataFile`.  The
base function is `independents()`, which automatically detects the type of the
independent parameter (and is available in the package namespace).  You can
manually call `frequencies()` and `times()` if this generation is not
successful.  For the dependent data, the main function is `probabilities()`,
also in the package namespace.
"""

import datetime
import numpy as np

__all__ = ['DataFile', 'independents', 'probabilities', 'times',
           'frequencies', 'metadata_fields']

def _nullable(parser):
    """
    Convert a parsing function into a 'nullable' function, i.e. one that returns
    `None` if the input is `'N/A'`, or parses it normally if not.
    """
    return lambda string: (None if string == 'N/A' else parser(string))

# Helper functions for converting various inputs.
_kHz = lambda f: float(f) * 2e3 * np.pi
_MHz = lambda f: float(f) * 2e6 * np.pi
_percent = lambda f: float(f) * 0.01

def _sideband(string: str):
    return {"0": 0, "R": -1, "B": 1}[string[-1]] * int(string[:-1])

# The fields in the metadata at the top of the file, with their in-file
# identifiers, the identifier we give them in Python, the parser used to
# convert their values, an identifier for the type after conversion, and a
# description of the field.
metadata_fields = [
    (
        'Spectroscopy data file',
        'time',
        lambda time: datetime.datetime.strptime(time, "%d/%m/%Y %H:%M:%S"),
        'datetime.datetime',
        'The time the spectrum was taken',
    ),
    (
        'Spectrum Type:',
        'type',
        str,
        'str = \'Continuous\' | \'Windowed\' | \'Fixed\'',
        'The type of spectrum taken',
    ),
    (
        '729 Direction:',
        'direction',
        str,
        'str = \'Axial\' | \'Radial\'',
        'The direction of the 729 laser',
    ),
    (
        'Trap Voltage (V):',
        'voltage',
        float,
        'float in V',
        'The trap voltage',
    ),
    (
        'Axial Frequency (kHz):',
        'axial',
        _kHz,
        'float in Hz',
        'The axial frequency',
    ),
    (
        'Modified Cyclotron Frequency (kHz):',
        'modified_cyclotron',
        _kHz,
        'float in Hz',
        'The modified cyclotron frequency',
    ),
    (
        'Magnetron Frequency (kHz):',
        'magnetron',
        _kHz,
        'float in Hz',
        'The magnetron frequency',
    ),
    (
        'AOM Start Frequency (MHz):',
        'aom_start',
        _MHz,
        'float in Hz',
        'The start frequency of the AOM',
    ),
    (
        'Carrier Frequency (MHz):',
        'carrier',
        _MHz,
        'float in Hz',
        'The frequency of the carrier',
    ),
    (
        'Step Size (kHz or ticks):',
        'step_size',
        float,
        'float in Hz | float in s',
        '''The spacing between two points in the scan.  If the scan is a
        frequency scan, this will be measured in Hz.  If it is a time scan, it
        will be in s.''',
    ),
    (
        'Sidebands to scan / side:',
        'total_sidebands',
        _nullable(lambda str_: 2 * int(str_) + 1),
        '?int',
        'The number of sidebands to scan over all files.',
    ),
    (
        'Sideband Width (steps):',
        'sideband_width',
        _nullable(int),
        '?int',
        'The number of steps taken over one sideband',
    ),
    (
        '729 RF Amplitude on profile 1 (%):',
        'amplitude',
        _percent,
        'float on [0, 1]',
        'The normalised RF amplitude of the 729 on profile 1.',
    ),
    (
        'Number of repeats per frequency:',
        'shots',
        int,
        'int',
        'The number of shots taken per point.',
    ),
    (
        'File contains interleaved spectra:',
        'total_spectra',
        int,
        'int',
        'The number of interleaved spectra in this file',
    ),
    (
        'This is sideband:',
        'sideband',
        _nullable(_sideband),
        '?int',
        'Which number sideband this is, if `total_sidebands` is not 1.',
    ),
    (
        'Pulse Start Length (fixed freq):',
        'start_time',
        _nullable(lambda x: int(x) * 40e-9),
        '?float in s',
        'The starting wait time in a fixed frequency scan.',
    ),
    (
        'Number of Steps (fixed freq):',
        'total_steps',
        _nullable(int),
        '?int',
        'The fixed frequency number of steps taken',
    ),
    (
        'Spectrum 1 name:',
        'name',
        str,
        'str',
        'The name of spectrum 1',
    ),
    (
        'Notes:',
        'notes',
        str,
        'str',
        'Any attached notes',
    ),
]

class DataFile:
    """
    Container class which holds an imported data file.  This includes all the
    metadata at the top of the file.  The data in here is still in the raw
    format that is output by the spectroscopy controller, so is not particularly
    useful for anything until one of the conversion functions has been called on
    it.

    Members --
        data: numpy.array(dtype=[("cool", "i4"), ("cool_error", "i4"),
                                 ("counts", "i4"), ("counts_error", "i4")]) --
            The raw data from the file, arranged into a structured numpy array.
            The different columns of data are can be accessed by indexing on the
            string identifying the columns, for example
                data['cool']
            or they can be accessed point-by-point as 4-tuples.
        file_name: str -- The file name that this class was read from.
        points: int -- The number of points in the spectrum.
    """
    def __init__(self, data, metadata, file_name):
        self.data = data
        self.file_name = file_name
        for detail, meta in zip(metadata_fields, metadata):
            setattr(self, detail[1], meta)
        if self.start_time is None:
            self.step_size = _kHz(self.step_size) # lin kHz to ang Hz
        else:
            self.step_size = self.step_size * 40e-9 # ticks to seconds
        self.points = self.data.shape[0] // self.shots

    def __repr__(self):
        preamble = f"Penning trap spectrum file '{self.file_name}':"
        attributes = "\n".join([f"  {attr}: {getattr(self, attr)}"
                                for _, attr, _, _, _ in metadata_fields])
        return "\n".join([preamble,
                          f"  points: {self.points}",
                          f"  step_size: {self.step_size}",
                          attributes])

DataFile.__doc__ = DataFile.__doc__.rstrip(" ")\
                   + "\n".join([f"        {attr}: {type} -- {desc}"
                                for _, attr, _, type, desc in metadata_fields])

def _points(data: np.array, shots: int) -> list:
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

def _spectra(data: np.array, spectra: int) -> list:
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

_doc_thresholds =\
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

def _point_probabilities(point, cool_threshold, count_thresholds, min_error):
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
_point_probabilities.__doc__ =\
    f"""
    Get the probabilities and errors of excitation for any number of ions, from
    a single point of data.

    Arguments --
    point: np.array(dtype=[("cool", "i4"), ("cool_error", "i4"),
                           ("counts", "i4"), ("counts_error", "i4")]) --
        A point of data.  Each row in the array corresponds to one shot from the
        point, and the structured fields are the same form as is read in when
        the data file is loaded.

    {_doc_thresholds.strip()}

    Returns --
    np.array(shape=(n_ions + 1,),
             dtype=[("probability", "f8"), ("error", "f8")]) --
        An array of the probabilities and errors of excitations.  The `n`th
        element of the output array is the probability and error that `n` ions
        were excited.  The sum of the probabilities in the array will always be
        equal to 1.
    """

def probabilities(data_file: DataFile,
                  cool_threshold: int,
                  count_thresholds: int,
                  min_error: float=0.01) -> np.array:
    return np.transpose(np.array(
        [_point_probabilities(point, cool_threshold, count_thresholds,min_error)
         for point in _points(data_file.data, data_file.shots)]))

probabilities.__doc__ =\
    f"""
    Get the probabilities and errors of excitation of any number of ions for all
    the points in the data file.

    Arguments --
    data_file: DataFile -- An output data file loaded using `penning.load`.
    {_doc_thresholds.strip()}

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
    try:
        return _independents[data_file.type](data_file)
    except KeyError:
        raise ValueError(f"Unknown scan type {data_file.type}.  Could not"
                         + " detect independent parameter type.")
