"""
Data analysis methods and tools for use with the Imperial College Penning trap
experiment.  This code is set up to work with the output of the Spectroscopy
Controller C# program, including parsing the metadata and converting the raw
output into useable data.

The basic function to load up a data file is `load()`, which returns a class
`DataFile`.  This contains the raw data as a structured `numpy.array` and the
metadata.

To extract the spectrum data, use `independents()` to get the automatically
detected independent parameters (frequency for frequency scans, time for Rabi
oscillations etc) as a `numpy.array`.  Use `probabilities()` to get a structured
`numpy.array` with the excitation probabilities for each number of ions and the
associated errors.

If the automatic detection fails, you can manually call
`data_file.frequencies()` or `data_file.times()` to suit your needs.

For fitting models and functions, look at the `fit` module.
"""

from .data_file import load, independents, probabilities, DataFile

from . import data_file, fit
