"""Support for loading data files from various sources into python.

This module contains the following classes:
    WMLTADataFile  -- wavemeter data file (.lta format)
    PMUtilDataFile -- desktop power meter utility data file
    PM100DDataFile -- handheld power meter data file

This module does NOT load spectrum data files. For that, see the spectrum.py
and loader.py modules.

The 'load' function allows for loading multiple data files and supports
relative paths, tilde expansion, and globbing, e.g.
    >>> from penning.datafile import *
    >>> dfs = load(PMUtilDataFile,
                   '~/Box Sync/Ion Trapping/PowerMeterData/2018-08-20**/*.txt')

Class initializers can also be used to load single data files but don't support
path expansion, e.g.
    >>> path = Path('~/Box Sync/Ion Trapping/PowerMeterData/'\
                    '2018-08-20_blue-power_pre-trap/blue1-axial.txt')
    >>> df = PMUtilDataFile(path.expanduser())

All loaded data is converted to un-prefixed units unless explicitly stated
otherwise (e.g. meters instead of nanometers, seconds instead of milliseconds, 
etc.).
"""
__all__ = ['load', 'WMLTADataFile', 'PMUtilDataFile', 'PM100DDataFile']

from pathlib import Path
from datetime import datetime
import numpy as np
import glob

# helper functions for reading a line of data
_readsplit = lambda file: file.readline().rstrip().split('\t')
_takelast = lambda file: _readsplit(file)[-1]

_unit = {'W' : 1,
         'ms': 1e-3,
         'mV': 1e-3,
         'µW': 1e-6,
         'nm': 1e-9}
# get unit string between [brackets] and return conversion factor
_parse_unit = lambda string: _unit[string.split('[')[1].split(']')[0]]

def load(cls, globpattern):
    """Initialize a 'cls' object for each file matching 'globpattern'.
    
    Arguments --
        cls: type -- name of class representing type of files to be loaded
        globpattern: str or pathlib.Path -- path to be expanded and searched
                                            for data files
    
    Returns --
        [cls] -- list of 'cls' objects loaded with data from each file matching
                 'globpattern'
    """
    return [cls(path) for path in
            glob.iglob(Path(globpattern).expanduser().as_posix())]

class WMLTADataFile:
    """A .lta file output from the HighFinesse WS8 wavemeter.
    
    According to the user manual, section 3.4 (p36): "The so stored lta (long
    term array) files...do not include raw measurement data, only the final
    results, it is not possible to conclude from lta file data to the
    measurement itself. The lta files are useful for further processings of the
    final results. To better be able to reanalyze your measurements with all
    its conditions and raw data, record the measurements to ltr files (longterm
    recording files by menu “Operation|Start|Recording...” in the Wavelength
    Meter main application."
    
    Members --
        path: pathlib.Path -- path to the data file on disk
        file_type: str     -- file type (1st line of file)
        file_version: int  -- file version (2nd line of file)
        nframes: int       -- number of frames in the file (see ._Frame class);
                              equivalent to len(self.frames)
        msrmnts: int       -- number of measurements in the file; equivalent
                              to the number of unique time bins in all frames
        start: datetime.datetime -- data acquisition start time
        end: datetime.datetime   -- data acquisition end time
        frames: [._Frame]   -- list of frames containing the data (see ._Frame
                              class)
    """
    
    class _Frame:
        """A single input/output channel of the HighFinesse WS8 wavemeter.
        
        Members --
            name: str -- channel name indicating the type of 'data'; e.g.
                         'Signal 1 Wavelength, vac.', or 'Analog output voltage
                         2', etc.
            times: np.ndarray -- measurement time bins in seconds
            data: np.ndarray -- wavemeter measurements in un-prefixed units
                                consistent with the 'name' attribute
        """
        def __init__(self, name, times, data):
            self.name = name
            self.times = times
            self.data = data
    
    def __init__(self, path):
        """Load data from the file at the given path.
        
        Arguments --
            path: str or pathlib.Path
        """
        self.path = Path(path)
        
        with open(self.path) as f:
            self.file_type = f.readline().rstrip()
            self.file_version = int(_takelast(f))
            f.readline()
            self.nframes = int(_takelast(f))
            self.msrmnts = int(_takelast(f))
            
            datetime_fmt = '%d.%m.%Y, %H:%M:%S.%f'
            self.start = datetime.strptime(_takelast(f), datetime_fmt)
            self.end = datetime.strptime(_takelast(f), datetime_fmt)
            
            # skip to [Measurement data] section
            while f.readline().rstrip() != '[Measurement data]' : pass
            f.readline()
            
            colnames = _readsplit(f)
            units = [_parse_unit(s) for s in colnames]
            frnames = [name.split('[')[0].rstrip() for name in colnames[1:]]
            
            # parse frame data and corresponding times
            frdata = [[] for _ in range(self.nframes)]
            frtimes = [[] for _ in range(self.nframes)]
            for line in f:
                readings = line.rstrip().split('\t')
                time = float(readings[0])
                for i, reading in enumerate(readings[1:]):
                    try:
                        frdata[i].append(float(reading))
                        frtimes[i].append(time)
                    except: pass
            
            # convert to numpy array with correct units
            frtimes = [np.array(times)*units[0] for times in frtimes]
            frdata = [np.array(fd)*u for fd, u in zip(frdata, units[1:])]
            self.frames = [WMLTADataFile._Frame(*params) for params
                           in zip(frnames, frtimes, frdata)]

class PMUtilDataFile:
    """
    A .txt file output from the ThorLabs optical power meter desktop utility.
    This is from the legacy PC software package.
    
    Members --
        path: pathlib.Path -- path to the data file on disk
        interface: dict    -- metadata about the sensor-PC interface
        sensor: dict       -- metadata about the sensor
        start: datetime.datetime -- data acquisition start time
        data: np.ndarray   -- power measurements in watts
        times: np.ndarray  -- measurement time bins, in units of seconds
                              relative to 'start'
    """
    
    def __init__(self, path):
        """Load data from the file at the given path.
        
        Arguments --
            path: str or pathlib.Path
        """
        self.path = Path(path)
        
        with open(self.path) as f:
            metadata = f.readline().split()
            self.interface = {'model': metadata[0],
                              'serial_number': metadata[1],
                              'firmware_version': metadata[3]}
            self.sensor = {'part_number': metadata[6],
                           'serial_number': metadata[7]}
            
            self.start, data0, unit_str = self._parse_line(f.readline())
            times = [0]
            data = [data0 * _unit[unit_str]]
            for line in f:
                timestamp, datum, unit_str = self._parse_line(line)
                times.append((timestamp - self.start).total_seconds())
                data.append(datum * _unit[unit_str])
            self.times = np.array(times)
            self.data = np.array(data)
    
    @staticmethod
    def _parse_line(line):
        """Split line at tabs and return (datetime, float, str)"""
        line = line.rstrip().split('\t')
        return (datetime.strptime(line[0], '%d/%m/%Y %H:%M:%S.%f '),
                float(line[1]), line[2])

class PM100DDataFile:
    """
    A file output from the ThorLabs PM100D handheld optical power meter console.
    
    Members --
        path: pathlib.Path -- path to the data file on disk
        sensor: dict       -- metadata about the sensor
        start: datetime.datetime -- data acquisition start time
        wavelength: str    -- wavelength setting (including units) during data
                              acquisition
        range: str         -- power range setting during data acquisition
        data: np.ndarray   -- power measurements in watts
        times: np.ndarray  -- measurement time bins in seconds relative to 'start'
    """
    
    def __init__(self, path):
        """Load data from the file at the given path.
        
        Arguments --
            path: str or pathlib.Path
        """
        self.path = Path(path)
        
        with open(path) as f:
            metadata = _readsplit(f)
            self.sensor = {'part_number': metadata[0]}
            self.start = datetime.strptime(metadata[1], '%Y-%m-%d %H:%M:%S')
            data_unit, time_unit = [_parse_unit(s) for s in _readsplit(f)]
            self.wavelength = _takelast(f)
            self.range = _takelast(f)
            
            data, times = [], []
            for line in f:
                line = line.split('\t')
                data.append(float(line[0]))
                times.append(float(line[1]))
            self.data = np.array(data) * data_unit
            self.times = np.array(times) * time_unit
