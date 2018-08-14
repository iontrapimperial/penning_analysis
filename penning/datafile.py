"""
Support for loading data files from various sources into python.

TODO: list classes and document usage
"""
__all__ = ['WMLTADataFile']

from pathlib import Path
from datetime import datetime
import numpy as np

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
            name: str -- channel name, including the units of 'data'; e.g.
                         'Signal 1 Wavelength, vac. [nm]', or 'Analog output
                         voltage 2 [mV]', etc.
            times: np.ndarray -- measurement time bins, in units of seconds
            data: np.ndarray -- wavemeter measurements, in units indicated by
                                the 'name' attribute
        """
        def __init__(self, name, times, data):
            self.name = name
            self.times = np.array(times)
            self.data = np.array(data)
    
    def __init__(self, path):
        """Load data from the file at the given path.
        
        Arguments --
            path: str or pathlib.Path
        """
        self.path = Path(path)
        
        with open(self.path) as f:
            read_next_line = lambda file: file.readline().rstrip().split('\t')
            read_next_value = lambda file: read_next_line(file)[-1]
            
            self.file_type = f.readline().rstrip()
            self.file_version = int(read_next_value(f))
            f.readline()
            self.nframes = int(read_next_value(f))
            self.msrmnts = int(read_next_value(f))
            
            datetime_fmt = '%d.%m.%Y, %H:%M:%S.%f'
            self.start = datetime.strptime(read_next_value(f), datetime_fmt)
            self.end = datetime.strptime(read_next_value(f), datetime_fmt)
            
            # skip to [Measurement data] section
            while f.readline().rstrip() != '[Measurement data]' : pass
            f.readline()
            
            frame_names = [name for name in read_next_line(f)][1:]
            frame_times = [[] for _ in range(self.nframes)]
            frame_data = [[] for _ in range(self.nframes)]
            for line in f:
                readings = line.rstrip().split('\t')
                time = float(readings[0])*1e-3 # convert ms -> s
                for i, reading in enumerate(readings[1:]):
                    try:
                        frame_data[i].append(float(reading))
                        frame_times[i].append(time)
                    except: pass
            self.frames = [WMLTADataFile._Frame(*params) for params in
                           zip(frame_names, frame_times, frame_data)]
