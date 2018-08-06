"""
Building blocks of pulse sequences.  There are two interfaces which can be used
to create the pulse sequences, a high(er) level one and a low level one
(where low-level just means it's as un-descriptive as the C#).  All times are in
seconds.

The high-level interface uses all the _functions_ defined in the module - see
their individual help for further uses.

The low-level interface means building up the pulse sequence using the _classes_
defined here.  This gives you more flexibility to do what you want, but you
don't get the descriptive names - you have to do everything yourself.
"""

import numpy as np
import numbers
import abc
from . import hex as hex_

__all__ = ['Pulse', 'Loop', 'Command',
           'doppler_cool', 'prepare_state', 'sideband_cool', 'probe', 'count',
           'send_data', 'wait', 'pause_point', 'change_frequency', 'loop']

def _in_set(laser):
    return lambda pulse: laser in pulse.lasers
def _dds_bit_unset(n):
    return lambda pulse: not pulse.dds_profile & (0x1 << n)

_laser_states = [
    ("Laser397B1", _in_set("b1")),
    ("Laser397B2", _in_set("b2")),
    ("Laser729", _in_set("729")),
    ("Laser854", _in_set("854")),
    ("Laser854POWER", _in_set("trap")),
    ("Laser854FREQ", _in_set("radial")),
    ("LaserAux1", lambda pulse: False),
    ("Laser729RF1", _dds_bit_unset(0)),
    ("Laser729RF2", _dds_bit_unset(1)),
    ("LaserAux2", _dds_bit_unset(2)),
]

_doc_required_members =\
    """
    name: str --
        Name of the pulse in the pulse creator.
    children: iterable of Element --
        Any children of this sequence element, in order.
    vars: set of str --
        The loose variable names that are present in this element, or any
        children of this element.
    shots: int --
        The number of shots of data that will be taken if this sequence element
        is run.  A shot is taken any time a "Count" operation happens, whether
        that is during the cooling phase, a dedicated counting phase or any
        other part of the experiment.
    static: bool --
        Whether the `xml()` method of this function is static, i.e. its result
        is independent of the arguments passed.  This is useful because static
        elements' XML representations can be cached, so they needn't be
        recreated multiple times in a looping construct.
    """

def _xml_command(element, time, type):
    ticks = int(round(time * 1e9 / 40)) # 1 tick = 40ns
    els = ['<Pulse']\
          + [f'{laser}="{"on" if state(element) else "off"}"'
             for laser, state in _laser_states]\
          + [f'Type="{type}"',
             f'Ticks="{ticks}"',
             f'TargetLength="{ticks * 0.04:.2f}"',
             f'Name="{element.name}"',
             '/>']
    return " ".join(els)

class Element(abc.ABC):
    """
    This abstract base class demonstrates what must be implemented by an element
    of the experimental sequence.  There are some required members, and a
    required `xml()` method to be implemented.

    Members --
    """
    def __init__(self):
        self.children = ()
        self.vars = set()
        self.shots = 0
        self.static = False
        self.name = "unnamed"

    @abc.abstractmethod
    def xml(self, args={}):
        """
        Return a string containing the XML representation of this sequence
        element.

        Arguments --
        args: dict of (key: str, val: *) --
            The context of variables which may be used within this element, or
            children of it.  The keys must be strings, which should minimally
            include all the variables used in lambda functions in this element
            or its descendents.  For example, if a property is bound as
                lambda t: 2 * t - 1
            then the `args` dictionary must include a `"t"` key.
        """
        raise NotImplementedError
Element.__doc__ = "\n".join([Element.__doc__,
                                     _doc_required_members.strip("\n")])

class Pulse(Element):
    """
    A pulse from any combination of the lasers for some (possibly variable)
    time.  The DDS profile which controls the 729 can be set as an integer, and
    the PMTs can be set to either count or not.

    Members --
    time: float in s | lambda --
        Either a fixed time, or a lambda function which, when its variables are
        defined by containing loop constructs, returns a time in seconds.

    count: bool --
        Whether the PMTs are counting during this pulse.

    lasers: set of ("b1" | "b2" | "729" | "854" | "trap" | "radial") --
        The lasers which are to be turned on during this pulse.  Here everything
        has its usual names, except `"trap"` refers to the pulse which causes
        the trap voltage to lower (referred to in the C# as "854 POWER"), and
        `"radial"` refers to the control of the radial 854 AOM ("854 FREQ" in
        C#), which removes the radial 854 during counting periods (allowing
        j-mixing to occur).

    dds_profile: int between 0 and 7 inclusive --
        Which number profile to set the DDS to during this pulse.
    """
    def __init__(self, time, lasers, dds_profile=0, count=False,
                 name="nameless"):
        self.time = time
        self.count = count
        self.lasers = set(lasers)
        self.dds_profile = dds_profile
        self.name = name
        if isinstance(self.time, numbers.Number):
            self.time = float(self.time)
            self.static = True
            self.vars = set()
        elif hasattr(self.time, "__call__"):
            self.vars = set(self.time.__code__.co_varnames)
            self.static = False
        else:
            raise TypeError("Can't interpret a time of type "\
                            + self.time.__class__.__name__ + ".")
        self.children = ()
        self.shots = int(self.count)

    def hex(self, args={}):
        time = self.time if isinstance(self.time, numbers.Number)\
               else self.time(*map(args.get, self.time.__code__.co_varnames))
        ticks = int(round(time * 1e9 / 40)) # 1 tick = 40ns
        type = hex_.CommandType.FIRE_LASERS_AND_COUNT if self.count\
               else hex_.CommandType.FIRE_LASERS
        return hex_.instruction(ticks, self.lasers, self.dds_profile, type)

    def xml(self, args={}):
        time = self.time if isinstance(self.time, numbers.Number)\
               else self.time(*map(args.get, self.time.__code__.co_varnames))
        return _xml_command(self, time, "Count" if self.count else "Normal")
Pulse.__doc__ =  "\n".join([Pulse.__doc__,
                            _doc_required_members.strip("\n")])

class Loop(Element):
    """
    A logical loop over one or more elements of a sequence.  This loop can be
    used for essentially any kind of `for` loop with a defined number of points,
    so it can be a "shots-per-point" loop, or a "frequency scan" or a "time
    scan", or really anything.

    Frequency loops must contain the a sequence element that corresponds to a
    frequency change (see `ChangeFrequency` or `change_frequency()`), because
    there must be an instruction which causes the FPGA to wait for computer
    control.

    Frequency loops and "shots-per-point" loops have a `loop_spec` which is just
    an integer, and these kinds of loops compile into "native" XML `<Loop>`
    instructions.

    More complex loops have a `str` as `loop_spec`, which is the name of the
    variable that will be looped.  To be useful, this variable should be the
    name of one of the loose variables contained in pulses within this loop.


    Members --

    loop_var: ?str --
        If this is a complex loop, then a string of the variable name that will
        be bound within the loop.  Otherwise undefined.

    loop_values: ?tuple --
        If this is a complex loop, then a tuple of all the values that the
        variable will take over the course of the loop.  Otherwise undefined.

    reps: int --
        How many times this loop will execute.

    unroll: bool --
        Whether this loop will be "unrolled" in the XML into a big long sequence
        of instructions.  If `False`, it will use a native `<Loop>` construct.

    fpga: bool --
        Whether this loop will be marked as an "FPGA loop".
    """
    def __init__(self, elements, loop_spec, loop_values=None, fpga=False,
                 name="nameless"):
        try:
            self.children = tuple(elements)
        except TypeError:
            self.children = (elements,)
        # prevent names from clashing inside loop (C# bug)
        for i, element in enumerate(self.children):
            element.name = f"({i+1:02d}) {element.name}"
        loose_names = set.union(*[el.vars for el in self.children])
        names_to_bind = set()
        if isinstance(loop_spec, str):
            names_to_bind = {loop_spec}
            if loop_values is None:
                raise ValueError("This loop needs to be unrolled, but I didn't"\
                                 + f" get any values to set '{loop_spec}' to.")
            self.loop_var = loop_spec
            self.loop_values = tuple(loop_values)
            self.reps = len(self.loop_values)
        elif isinstance(loop_spec, int):
            self.reps = loop_spec
        else:
            raise TypeError(" ".join([
                f"Unexpected type of `loop_spec` '{type(loop_spec).__name__}'.",
                "This should be either an integer number of repetitions, or",
                "a string of a variable name to be bound inside the loop (and",
                "the next argument should be its values)."]))
        self.static = not loose_names
        self.unroll = bool(names_to_bind)
        self.vars = loose_names - names_to_bind
        self.shots = sum(map(lambda e: e.shots, self.children)) * self.reps
        self.fpga = fpga
        self.name = name

    def hex(self, args={}):
        if self.fpga:
            loop_start = hex_.instruction(self.reps, set(), 6,
                                         hex_.CommandType.LOOP_START)
            content = b"".join([el.hex(args) for el in self.children])
            loop_end = hex_.instruction(0, set(), 6, hex_.CommandType.LOOP_END)
            return b"".join([loop_start, content, loop_end])
        elif not self.unroll:
            return b"".join([el.hex(args) for el in self.children]) * self.reps
        else:
            args = args.copy()
            vary_indices = [i for i, el in enumerate(self.children)\
                            if not el.static]
            # Make it always go static - vary - static - vary - static all
            # the way along (though some static bits might be empty string).
            statics = []
            prev = 0
            for i in vary_indices:
                statics.append(b"".join([el.hex()\
                                        for el in self.children[prev:i]]))
                prev = i + 1
            statics.append(b"".join([el.hex() for el\
                                    in self.children[vary_indices[-1] + 1:]]))
            def per_loop(value):
                args[self.loop_var] = value
                for i, vary in enumerate(vary_indices):
                    yield statics[i]
                    yield self.children[vary].hex(args)
                yield statics[-1]
            return b"".join((b"".join(per_loop(value))\
                            for value in self.loop_values))

    def xml(self, args={}):
        if not self.unroll:
            loop_head = " ".join(['<Loop',
                                  f'LoopCount="{self.reps}"',
                                  f'Name="{self.name}"',
                                  f'FPGALoop="{self.fpga}">'])
            content = "".join([el.xml(args) for el in self.children])
            return "".join([loop_head, content, "</Loop>"])
        else:
            args = args.copy()
            vary_indices = [i for i, el in enumerate(self.children)\
                            if not el.static]
            # Make it always go static - vary - static - vary - static all
            # the way along (though some static bits might be empty string).
            statics = []
            prev = 0
            for i in vary_indices:
                statics.append("".join([el.xml()\
                                        for el in self.children[prev:i]]))
                prev = i + 1
            statics.append("".join([el.xml() for el\
                                    in self.children[vary_indices[-1] + 1:]]))
            def per_loop(value):
                args[self.loop_var] = value
                for i, vary in enumerate(vary_indices):
                    yield statics[i]
                    yield self.children[vary].xml(args)
                yield statics[-1]
            return "".join(("".join(per_loop(value))\
                            for value in self.loop_values))
Loop.__doc__ =  "\n".join([Loop.__doc__,
                           _doc_required_members.strip("\n")])

_command_type = {
        "SendData": hex_.CommandType.SEND_DATA,
        "Wait_Labview": hex_.CommandType.WAIT_FOR_COMPUTER,
}

class Command(Element):
    """
    A sequence command which has no interacting lasers, takes no time, but only
    performs a command (e.g. changing the frequency, sending data, etc).

    Members --
    type: str --
        The XML `Type` attribute.  These are defined by the C# code.
    """
    def __init__(self, type, name="unnamed"):
        self.static = True
        self.vars = ()
        self.children = ()
        self.shots = 0
        self.name = name
        self.type = type
        self.lasers = set(["b1", "b2", "854", "radial"])
        self.dds_profile = 0
    def hex(self, args={}):
        return hex_.instruction(0, self.lasers, 0, _command_type[self.type])
    def xml(self, args={}):
        return _xml_command(self, 0, self.type)
Command.__doc__ =  "\n".join([Command.__doc__,
                              _doc_required_members.strip("\n")])

def doppler_cool(time=10e-3, count=True, name="Doppler cool"):
    """
    A Doppler cooling pulse, which has blue 1, blue 2, the 854 and the radial
    854 lasers active, and counts for its duration.
    """
    return Pulse(time, ["b1", "b2", "854", "radial"], count=count,
                 name=name)
def prepare_state(time=100e-6):
    """
    The ground state pumping beams, with blue 1, the 854 and the radial 854
    active.
    """
    return Pulse(time, ["b1", "854", "radial"], name="Prepare state")

def sideband_cool(time=10e-3, profile=1, name="Sideband cool"):
    """
    A sideband cooling pulse, which has blue 1, the 854 and the radial
    854 lasers active, the 729 active on the specified profile (defaults to
    profile 1 if not given) and does _not_ count during.
    """
    return Pulse(time, ["b1", "729", "854", "radial"], dds_profile=profile,
                 name=name)

def probe(time, profile, add_lasers=[], name="Probe"):
    """
    A pulse using only the 729 laser on a given DDS profile.  This is the
    bread-and-butter state manipulation pulse.
    """
    return Pulse(time, ["729"] + add_lasers, dds_profile=profile, name=name)

def wait(time, profile=0, name="Wait"):
    """
    Wait for a time with no lasers interacting.  The DDS profile can be set, but
    the 729 is not interacting with the ion.
    """
    return Pulse(time, [], dds_profile=profile, name=name)

def count(time=10e-3, name="Count"):
    """
    Perform a count operation.  This has lasers blue 1 and blue 2 active.
    """
    return Pulse(time, ["b1", "b2"], count=True, name=name)

def send_data(name="Send data"):
    """
    Issue a command to transfer the data on the FPGA to the computer.
    """
    return Command("SendData", name=name)

def pause_point(name="Pause point"):
    """
     Insert a place where the experiment can be paused in a fixed frequency
     scan.

     NOTE: due to C# limitations, this is equivalent to the "change frequency"
     command, so it will have that unintended side-effect if not used in "Fixed"
     spectrum mode.
     """
    return Command("Wait_Labview", name=name)

def change_frequency(name="Change frequency"):
    """
    Insert an explicit command to change frequency.  This has no effect if in
    "Fixed" mode, except the experiment will be able to be paused at that
    point.
    """
    return Command("Wait_Labview", name=name)

def loop(elements, loop_spec, loop_var=None, fpga=False, name="Loop"):
    """
    A logical loop over one or more elements of a sequence.  This loop can be
    used for essentially any kind of `for` loop with a defined number of points,
    so it can be a "shots-per-point" loop, or a "frequency scan" or a "time
    scan", or really anything.

    Frequency loops must contain the a sequence element that corresponds to a
    frequency change (see `ChangeFrequency` or `change_frequency()`), because
    there must be an instruction which causes the FPGA to wait for computer
    control.

    Frequency loops and "shots-per-point" loops have a `loop_spec` which is just
    an integer, and these kinds of loops compile into "native" XML `<Loop>`
    instructions.

    More complex loops have a `str` as `loop_spec`, which is the name of the
    variable that will be looped.  To be useful, this variable should be the
    name of one of the loose variables contained in pulses within this loop.

    Arguments --
    elements: Element or iterable of Element --
        The children of the loop, in order.

    loop_var: ?str --
        If this is a complex loop, then a string of the variable name that will
        be bound within the loop.  Otherwise undefined.

    loop_values: ?tuple --
        If this is a complex loop, then a tuple of all the values that the
        variable will take over the course of the loop.  Otherwise undefined.

    fpga: ?bool --
        Whether to compile this using an FPGA loop instruction or not.

    name: ?str --
        The name to give this loop in the pulse creator.  Has no effect if this
        is a complex loop, since it will be unrolled.
    """
    return Loop(elements, loop_spec, loop_var, fpga, name)
