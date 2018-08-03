"""
Functions for creating XML files that can be manipulated by the Spectroscopy
Controller's pulse sequence designer.  The writeout is performed by `write()`,
and the building blocks of the pulse sequences are in the `elements` module.
Typically you might want to do `from elements import *` - this will only put the
building block elements in your global namespace.
"""

from .api import *
from . import elements
from . import api as _api

__all__ = ['elements'] + _api.__all__
