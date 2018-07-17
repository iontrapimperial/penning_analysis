"""
Contains the `load` function, which loads up an output file from the
spectroscopy controller, including parsing all of the metadata.  This outputs a
class `DataFile`, which is a Python representation of the data and the metadata.
"""

import datetime, collections
import numpy as np

__all__ = ['load', 'DataFile']

def _nullable(parser):
    """
    Convert a parsing function into a 'nullable' function, i.e. one that returns
    `None` if the input is `'N/A'`, or parses it normally if not.
    """
    return lambda string: (None if string == 'N/A' else parser(string))

# Helper functions for converting various inputs.
_kHz = lambda f: float(f) * 1e3
_MHz = lambda f: float(f) * 1e6
_percent = lambda f: float(f) * 0.01

def _sideband(string: str):
    return {"0": 0, "R": -1, "B": 1}[string[-1]] * int(string[:-1])

# The fields in the metadata at the top of the file, with their in-file
# identifiers, the identifier we give them in Python, the parser used to
# convert their values, an identifier for the type after conversion, and a
# description of the field.
_metadata_fields = [
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
        'str = \'Continuous\' | ?',
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
        fixed_frequency: bool --
            Whether this scan was a fixed frequency scan (e.g. a scan of wait
            times) or not.
    """
    def __init__(self, data, metadata, file_name):
        self.data = data
        self.file_name = file_name
        for detail, meta in zip(_metadata_fields, metadata):
            setattr(self, detail[1], meta)
        if self.start_time is None:
            self.fixed_frequency = False
            self.step_size = self.step_size * 1e3 # kHz to Hz
        else:
            self.fixed_frequency = True
            self.step_size = self.step_size * 40e-9 # ticks to seconds
        self.points = self.data.shape[0] // self.shots

    def __repr__(self):
        preamble = f"Penning trap spectrum file '{self.file_name}':"
        attributes = "\n".join([f"  {attr}: {getattr(self, attr)}"
                                for _, attr, _, _, _ in _metadata_fields])
        return "\n".join([preamble,
                          f"  points: {self.points}",
                          f"  step_size: {self.step_size}",
                          attributes])

DataFile.__doc__ = DataFile.__doc__.rstrip(" ")\
                   + "\n".join([f"        {attr}: {type} -- {desc}"
                                for _, attr, _, type, desc in _metadata_fields])

def load(file: str) -> DataFile:
    """
    Given a path to a data file, read the metadata and data into Python types
    and return the resulting class.
    """
    metadata = []
    with open(file, "r") as f:
        line = 0
        for description, _, parse, type_, _ in _metadata_fields:
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
                    names='cool, cool_error, counts, counts_error',
                    formats='i4, i4, i4, i4')
    return DataFile(data, metadata, file)
