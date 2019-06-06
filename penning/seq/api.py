"""
Provides the top-level commands to the module for creating and writing XML and
FPGA hex files.  These are `create_xml()`, `create_hex()`, `write_xml()` and
`write_hex()` respectively.  The XML files are only still available for
compatability with the C# code.  The hex files are all that is actually needed.
"""

from . import fpga

__all__ = ['create_xml', 'write_xml', 'create_hex', 'write_hex']

_XML_PREAMBLE = '<?xml version="1.0" encoding="utf-8"?><Experiment>'
_XML_EPILOGUE = '<Pulse Laser397B1="off" Laser397B2="off" Laser729="off" Laser854="off" Laser729RF1="off" Laser729RF2="off" Laser854POWER="off" Laser854FREQ="off" LaserAux1="off" LaserAux2="off" Type="Stop" Ticks="0" TargetLength="0" Name="Stop" /></Experiment>'

def create_xml(elements, base_args=None):
    """
    Make a string of XML representing the pulse sequence `elements`.

    Arguments --
    elements: seq.Element or iterable of seq.Element --
        The elements in the sequence.  This may often just be a single `Loop`
        element.
    base_args: dict --
        The base argument dictionary to use, which must fill in any loose
        variables which are not defined by loops.
    """
    if base_args is None:
        base_args = {}
    try:
        body = "".join([e.xml(base_args) for e in elements])
    except TypeError:
        body = elements.xml(base_args)
    return "".join([_XML_PREAMBLE, body, _XML_EPILOGUE])

def write_xml(file, elements, base_args=None):
    """
    Write out an XML file which can be loaded by the Spectroscopy Controller
    pulse sequence designer.

    Arguments --
    file: str --
        The file name to write out to.
    elements: seq.Element or iterable of seq.Element --
        The elements in the sequence.  This may often just be a single `Loop`
        element.
    base_args: dict --
        The base argument dictionary to use, which must fill in any loose
        variables which are not defined by loops.
    """
    if base_args is None:
        base_args = {}
    str_ = create_xml(elements, base_args)
    with open(file, "w") as f:
        print(str_, file=f)

def create_hex(elements, base_args=None):
    """
    Make a bytestring of FPGA hex representing the pulse sequence `elements`.

    Arguments --
    elements: seq.Element or iterable of seq.Element --
        The elements in the sequence.  This may often just be a single `Loop`
        element.
    base_args: dict --
        The base argument dictionary to use, which must fill in any loose
        variables which are not defined by loops.
    """
    if base_args is None:
        base_args = {}
    try:
        body = b"".join([e.hex(base_args) for e in elements])
    except TypeError:
        body = elements.hex(base_args)
    return body + fpga.STOP

def write_hex(file, elements, base_args=None):
    if base_args is None:
        base_args = {}
    """
    Write out an FPGA hex file which can be directly uploaded to the FPGA.

    Arguments --
    file: str --
        The file name to write out to.
    elements: seq.Element or iterable of seq.Element --
        The elements in the sequence.  This may often just be a single `Loop`
        element.
    base_args: dict --
        The base argument dictionary to use, which must fill in any loose
        variables which are not defined by loops.
    """
    with open(file, "wb") as f:
        f.write(create_hex(elements, base_args))
