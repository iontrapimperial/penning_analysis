"""
Data analysis methods and tools for use with the Imperial College Penning trap
experiment.  This code is set up to work with the output of the Spectroscopy
Controller C# program, including parsing the metadata and converting the raw
output into useable data.
"""

from .load import load, DataFile
