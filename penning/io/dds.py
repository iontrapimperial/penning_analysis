"""
Control of and communication with the DDS Arduino (and so indirectly the DDS).
This includes the structure of a DDS profile (`Profile`), and the functions to
interface with the Arduino - `open()`, `close()`, `set()` and `reset()`.  This
uses the `serial` module underneath, accessible on `conda` as `pyserial`.
"""

import numpy as np
import struct
import collections
import serial

__all__ = ['Profile', 'open', 'close', 'set', 'reset']

Profile = collections.namedtuple('Profile', ['frequency', 'amplitude', 'phase'])
Profile.__doc__ =\
    """
    A single DDS profile.  This has a frequency (angular Hz), an amplitude
    ([0, 1]) and a phase ([0, 2*pi]).
    """
Profile.frequency.__doc__ = "The frequency to set to as a float in linear Hz."
Profile.amplitude.__doc__ = "The relative amplitude as a float on [0, 1]"
Profile.phase.__doc__ = "The phase shift at the start as a float in radians."

_2_bytes = struct.Struct(">H")
_4_bytes = struct.Struct(">L")

def _ascii_numerals(byte):
    """
    Convert a byte into a byte string of the ascii numerals.  For example,
        224 -> b"224"
    and so on.
    """
    return str(byte).encode('ascii')

# Scale factor to map the range 0-1 linear GHz onto 4 bytes.
_ftw_scale = 0xffff_ffff / (1e9 * 2*np.pi)
def _ftw(frequency_offset):
    """
    Get the "frequency tuning word" (FTW) from the DDS profile.  Returns as a
    comma-separated byte string of the ASCII representation of bytes in
    big-endian order.
    """
    ftw = int(np.round(frequency_offset * _ftw_scale))
    return b",".join(map(_ascii_numerals, _4_bytes.pack(ftw)))

# Scale factor to map the range [0, 2pi] onto 2 bytes.
_pow_scale = 0xffff / (2*np.pi)
def _pow(phase):
    """
    Get the "phase offset word" (POW) from the DDS profile.  Returns as a
    comma-separated byte string of the ASCII representation of bytes in
    big-endian order.
    """
    pow = int(np.round(phase * _pow_scale))
    return b",".join(map(_ascii_numerals, _2_bytes.pack(pow)))

def _poly(coefficients, x):
    """
    Calculate the value of a polynomal in `x`, where the `coefficients` are in
    "reverse" order.  For example, if we have `c = coefficients`, then this
    returns
        c[0] * x**n + c[1] * x**(n-1) + ... c[n-1] * x + c[n],
    so if there are `m` coefficients in the array, then this will return the
    value of an `m-1`-order polynomial in `x`.
    """
    out = coefficients[0]
    for coefficient in coefficients[1:]:
        out = out * x + coefficient
    return out

def _relevant_poly(thresholds, coefficients, x):
    """
    Choose the relevant polynomial coefficients based on some threshold values.
    If `x` falls between `thresholds[i]` and `thresholds[i+1]`, then return the
    polynomial in `x` defined by `coefficients[i]`.

    Raises `ValueError` if `x` is outside of the range of the thresholds.
    """
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= x <= thresholds[i + 1]:
            return _poly(coefficients[i], x)
    raise ValueError("Parameter out of range for amplitude fitting.")

# The actual amplitude output of the AOMs depends on the frequency that they're
# being asked to emit.  To try to normalise the output of the AOM such that the
# amplitudes are consistent between frequencies, there are several fitting
# regimes which characterise the scale factor that should be applied.
#
# The thresholds for fitting the frequencies are all given in linear MHz, as are
# the coefficients of the cubic polynomial mapping.  The fitted polynomial gives
# the "root-mean-square" (of something I'm not sure about), which is essentially
# a scaling factor.  Vince's PhD thesis has more information.  The coefficients
# were found by fitting data on 2016-10-25, and the C# has the comment:
#   Measured with ASF = 8181 and 30 dB attenuation on a Tektronix MSO3034 scope.
#   Measurements done on 25/10/2016. See Vince's lab book for details.
_asf_frequency_thresholds = [100, 120, 180, 230, 300]
_asf_rms_coefficients = np.array([
    [0, 0.02, -4.5, 595],
    [2.3543e-4, -0.10083, 13.333, -211.91],
    [1.7716e-4, -0.10867, 21.385, -1067.2],
    [-6.1763e-5, 0.037394, -7.3432, 718.86],
])
# These thresholds are similar to those for the frequency, but are only used to
# "normalise" the amplitudes that the user inputs into roughly what is actually
# output.
_asf_amplitude_thresholds = [0, 0.05, 0.9156, 1]
_asf_amplitude_coefficients = np.array([
    [0, 0, 1, 0],
    [0.878114279, -1.323317, 1.26421389, 0.0775117],
    [461.13, -1305.0, 1232.3, -387.44],
])
# Scale factor for the amplitude.  The 0.7 is to prevent saturation of the AOM,
# the 217 is 217mV and is the minimum response that was measured in the
# 100--300MHz range.
_asf_scale = 0.7 * 217 * ((1 << 14) - 1)

def _asf(frequency_offset, amplitude, normalise):
    """
    Get the "amplitude scale factor" (ASF) from the DDS profile.  Returns as a
    comma-separated byte string of the ASCII representation of bytes in
    big-endian order.  The amplitude response of the DDS is highly non-linear in
    frequency and the base "ASF".  This code will normalise with respect to
    frequency always, but will only linearise the `amplitude` parameter if
    `normalise`.

    Arguments --
    frequency_offset: float in anuglar Hz -- The targetted frequency offset.
    amplitude: float on [0, 1] --
        The relative amplitude to set.  This is not a linear scale unless
        `normalise` is given as `True`.
    normalise: bool --
        Whether to linearise the amplitude scale (so 0.5 is half amplitude
        compared to 1.0).
    """
    # This is essentially a scaling factor.  The fitted polynomials have
    # coefficients corresponding to a frequnecy in linear MHz.
    rms = _relevant_poly(_asf_frequency_thresholds, _asf_rms_coefficients,
                         frequency_offset / (2e6 * np.pi))
    asf = amplitude * _asf_scale / rms
    if normalise:
        asf = asf * _relevant_poly(_asf_amplitude_thresholds,
                                   _asf_amplitude_coefficients,
                                   amplitude)
    asf = int(np.round(asf))
    return b",".join(map(_ascii_numerals, _2_bytes.pack(asf)))

def _message(profiles, normalise):
    """
    Create the bytestring message that should be send to the DDS Arduino to set
    all 8 profiles.

    Arguments --
    profiles: iterable of Profile -- The 8 DDS profiles to set.
    normalise: bool -- Whether `Profile.amplitude` should be linearised.
    """
    def single(profile):
        return b",".join([_asf(profile.frequency, profile.amplitude, normalise),
                         _pow(profile.phase),
                         _ftw(profile.frequency)])
    return b",".join(map(single, profiles))

def open(address, timeout=5):
    """
    Open the DDS Arduino at `address` with the specified `timeout` in seconds
    (can be float).  The output of this can be used as a context manager (i.e.
    within a `with` block) to automatically handle the closing, or you can call
    the `dds.close()` function on it.

    Arguments --
    address: str --
        The name of the serial port to access the DDS Arduino on.  For Windows,
        this will be something like "COM12".  For Mac it will be something like
        "/dev/tty.xxxxx", and for Linux it will probably be like "/dev/ttyxxx".
    timeout: ?float --
        The time to wait in seconds before a read operation times out.

    Returns --
    device: serial.Serial -- A serial interface to the DDS.
    """
    return serial.Serial(address, baudrate=9600, timeout=timeout)

def _write(dds, message, terminate=True):
    """
    Low-level write command to the DDS.  Writes the given bytestring message,
    then checks the error code response.
    """
    message = message + (b"\n" if terminate else b"")
    dds.write(message)
    response = dds.read_until(b"\n")
    try:
        error_code = int(response)
    except ValueError:
        raise ConnectionError("Expected an integer error code as a DDS"\
                              + f" response, but got: {response}") from None
    if error_code is not 1:
        raise ConnectionError("Got a failure code back from the DDS."\
                              + f"  Expected to get 1, but got {error_code}.")

# There's an effective resolution on our ability to set the DDS frequency.  This
# epsilon (1 lin Hz) is the allowed variation between what we tried to set a
# profile to, and what it actually got set to.
_error_epsilon = 2*np.pi
# When we set the profiles, the DDS Arduino responds with the set values of a
# select number of profiles.  These are the indices of the ones it uses.
_check_frequencies = [0, 5, 7]
def set(dds, profiles, normalise=False):
    """
    Set the DDS `dds` to have the profiles `profiles`.  There should be exactly
    8 profiles, and they should come in numerical order in the iterable.  It is
    a `ValueError` to give an incorrect number of profiles.

    Arguments --
    dds: serial.Serial --
        The `Serial` class for the DDS Arduino.  Typically you would get this as
        the output of `dds.open()`.
    profiles: iterable of `dds.Profile` -- The 8 profiles to use in order.
    normalise: ?bool -- Whether to linearise the amplitude parameter.

    Raises --
    ValueError -- If the number of profiles isn't 8.
    ConnectionError --
        If communication with the DDS failed in some manner, or if the DDS
        Arduino gave back an unexpected response.  These two issues indicate a
        failure in the Python/DDS Arduino logic.
    """
    profiles = tuple(profiles)
    if len(profiles) != 8:
        raise ValueError(f"Need to set 8 profiles, but got {len(profiles)}.")
    _write(dds, _message(profiles, normalise))
    responses = [dds.read_until() for _ in _check_frequencies]
    try:
        dds_frequencies = [int(response) / _ftw_scale for response in responses]
    except ValueError:
        raise ConnectionError(f"Expected to get {len(responses)} check"\
                              + " frequencies back from the DDS as integers,"\
                              + f" but instead got: {responses}.") from None
    for i, dds_f in enumerate(dds_frequencies):
        which = _check_frequencies[i]
        attempt_f = profiles[which].frequency
        if abs(dds_f - attempt_f) > _error_epsilon:
            raise ConnectionError(f"DDS Profile {which} was not set correctly."\
                                  + f"  Tried to set to {attempt_f}, but"\
                                  + f" instead set to {dds_f}.")
    return

# The message that causes the DDS Arduino to reset.
_reset_message = b",".join([b"256"] * 64)
def reset(dds):
    """
    Send the reset signal to the DDS Arduino.  Communications will still be open
    after this occurs.
    """
    _write(dds, _reset_message)

def close(dds):
    """
    Close the connection to the DDS Arduino.  If you were using the DDS as part
    of a `with` block, this command is unnecessary.
    """
    dds.close()
