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
from . import fpga

__all__ = ['Pulse', 'Loop',
           'doppler_cool', 'pump_to_ground', 'sideband_cool', 'probe', 'count',
           'send_data', 'wait', 'pause_point', 'change_frequency', 'loop']

def _in_set(laser):
    return lambda pulse: laser in pulse.lasers
def _dds_bit_unset(n):
    return lambda pulse: not pulse.dds & (0x1 << n)

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
    vars: set of str --
        The loose variable names that are present in this element, or any
        children of this element.
    shots: int --
        The number of shots of data that will be taken if this sequence element
        is run.  A shot is taken any time a "Count" operation happens, whether
        that is during the cooling phase, a dedicated counting phase or any
        other part of the experiment.
    """

def _ticks(time):
    return int(round(time * 1e9 / 40))

class Element(abc.ABC):
    """
    This abstract base class demonstrates what must be implemented by an element
    of the experimental sequence.  There are some required members, and a
    required `xml()` method to be implemented.

    Members --
    """
    def __init__(self):
        self.vars = set()
        self.shots = 0
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

    @abc.abstractmethod
    def hex(self, args={}):
        """
        Return a bytestring of the FPGA's hex file for this element.

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
    def __init__(self, time, lasers, dds, type, name="nameless"):
        self.__time = time
        self.type = type
        self.shots = int(self.type is fpga.FIRE_LASERS_AND_COUNT)
        self.lasers = set(lasers)
        self.dds = dds
        self.name = name
        if isinstance(self.__time, numbers.Number):
            self.__time = float(self.__time)
            self.vars = set()
            self.time = lambda args={}: self.__time
            self.static = True
        elif hasattr(self.__time, "__call__"):
            self.vars = set(self.__time.__code__.co_varnames)
            self.time = self.__call_time
            self.static = False
        else:
            raise TypeError("Can't interpret a time of type "\
                            + self.time.__class__.__name__ + ".")

    def __call_time(self, args):
        """
        Extract the actual desired time from a callable `time`.
        """
        return self.__time(*map(args.get, self.__time.__code__.co_varnames))

    def hex(self, args={}):
        ticks = _ticks(self.time(args))
        return fpga.instruction(ticks, self.lasers, self.dds, self.type)

    def xml(self, args={}):
        ticks = _ticks(self.time(args))
        els = ['<Pulse']\
              + [f'{laser}="{"on" if state(self) else "off"}"'
                 for laser, state in _laser_states]\
              + [f'Type="{self.type.xml}"',
                 f'Ticks="{ticks}"',
                 f'TargetLength="{ticks * 0.04:.2f}"',
                 f'Name="{self.name}"',
                 '/>']
        return " ".join(els)
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
        # prevent names from clashing inside loop (C# bug workaround)
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
        self.unroll = bool(names_to_bind)
        self.vars = loose_names - names_to_bind
        self.shots = sum(map(lambda e: e.shots, self.children)) * self.reps
        self.fpga = fpga
        self.name = name
        if self.unroll and self.fpga:
            raise ValueError("This loop needs to be unrolled because it is"\
                             + f" meant to bind variables ({names_to_bind}),"\
                             + " so it cannot also be an FPGA loop.")

    def __split_statics_varies(self, base, create):
        statics = []
        varies = [i for i, el in enumerate(self.children) if el.vars]
        prev = 0
        # Make it always go static - vary - static - vary - static all the way
        # along (though some static bits might be empty string).
        for i in varies:
            statics.append(base.join([getattr(el, create)()\
                                     for el in self.children[prev:i]]))
            prev = i + 1
        statics.append(base.join([getattr(el, create)()\
                                 for el in self.children[varies[-1] + 1:]]))
        return statics, varies

    def __looper(self, args, base, create):
        statics, varies = self.__split_statics_varies(base, create)
        def loop(value):
            args[self.loop_var] = value
            for i, vary in enumerate(varies):
                yield statics[i]
                yield getattr(self.children[vary], create)(args)
            yield statics[-1] # no chance of repeat because len(static) >= 2
        return loop

    def hex(self, args={}):
        if self.fpga:
            loop_start = fpga.instruction(self.reps, set(), 6, fpga.LOOP_START)
            content = b"".join([el.hex(args) for el in self.children])
            loop_end = fpga.instruction(0, set(), 6, fpga.LOOP_END)
            return b"".join([loop_start, content, loop_end])
        elif not self.unroll:
            return b"".join([el.hex(args) for el in self.children]) * self.reps
        else:
            per_loop = self.__looper(args.copy(), b"", "hex")
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
            per_loop = self.__looper(args.copy(), "", "xml")
            return "".join(("".join(per_loop(value))\
                            for value in self.loop_values))
Loop.__doc__ =  "\n".join([Loop.__doc__,
                           _doc_required_members.strip("\n")])

def _pulse_preset(lasers, time, count, dds, name):
    def _pulse(time=time, count=count, dds=dds, name=name):
        return Pulse(time, lasers, dds,
                     fpga.FIRE_LASERS_AND_COUNT if count else fpga.FIRE_LASERS,
                     name)
    _pulse.__doc__ = f"Active lasers: {lasers}"
    return _pulse

doppler_cool = _pulse_preset(["b1", "b2", "854", "radial"], 10e-3, True, 0,
                             "Doppler cool")
sideband_cool = _pulse_preset(["b1", "729", "854", "radial"], 10e-3, False, 1,
                              "Sideband cool")
pump_to_ground = _pulse_preset(["b1", "854", "radial"], 100e-6, False, 0,
                               "Prepare state")
probe = _pulse_preset(["729"], 0, False, 0, "Probe")
wait = _pulse_preset([], 0, False, 0, "Wait")
count = _pulse_preset(["b1", "b2"], 10e-3, True, 0, "Count")

def send_data(name="Send data"):
    """
    Issue a command to transfer the data on the FPGA to the computer.
    """
    return Pulse(0, ["b1", "b2", "854", "radial"], 0, fpga.SEND_DATA, name=name)

def pause_point(name="Pause point"):
    """
     Insert a place where the experiment can be paused in a fixed frequency
     scan.

     NOTE: due to C# limitations, this is equivalent to the "change frequency"
     command, so it will have that unintended side-effect if not used in "Fixed"
     spectrum mode.
     """
    return Pulse(0, ["b1", "b2", "854", "radial"], 0, fpga.WAIT_FOR_COMPUTER,
                 name=name)

def change_frequency(name="Change frequency"):
    """
    Insert an explicit command to change frequency.  This has no effect if in
    "Fixed" mode, except the experiment will be able to be paused at that
    point.
    """
    return Pulse(0, ["b1", "b2", "854", "radial"], 0, fpga.WAIT_FOR_COMPUTER,
                 name=name)

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
