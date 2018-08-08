"""
Module for creation of sequences of pulses for experiments.  This module can
write out XML files for viewing the sequences in the Spectroscopy Contoller, but
more importantly it can directly write out FPGA hex files (though these still
need to be uploaded).

The creation and file writing functions are `create_{}()` and `write_{}()`
respectively, where the `{}` can be either `xml` or `hex`.

The building blocks of the pulse sequences are in the `elements` module, where
more help is available.  Typically you might want to do `from elements import *`
- this will only put the building block elements in your global namespace.
"""

from .api import *
from . import elements
from . import api as _api

__all__ = ['elements'] + _api.__all__
