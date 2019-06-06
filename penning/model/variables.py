import numpy as np

def laguerre(n: int, a: float, x: float) -> float:
    """laguerre(n : int >= 0, a : float, x : float) -> res : float

    Calculate the Laguerre polynomial result L_n^a(x), which is equivalent to
    Mathematica's LaguerreL[n, a, x].
    """
    if n == 0:
        return 1
    elif n == 1:
        return 1 + a - x
    # use a recurrence relation calculation for speed and accuracy
    # ref: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    l_2, l_1 = 1, 1 + a - x
    for m in range(2, n + 1):
        l_2, l_1 = l_1, ((a + 2*m - x - 1) * l_1 - (a + m - 1) * l_2) / m
    return l_1

def laguerre_range(n_start: int, n_end: int, a: float, x: float) -> np.ndarray:
    """laguerre_range(n_start, n_end, a, x) -> np.array(dtype=np.float64)

    Use the recurrence relation for nearest-neighbour in n of the Laguerre
    polynomials to calculate
        [laguerre(n_start, a, x),
         laguerre(n_start + 1, a, x),
         ...,
         laguerre(n_end - 1, a, x)]
    in linear time of `n_end` rather than quadratic.

    The time is linear in `n_end` not in the difference, because the initial
    calculation of `laguerre(n_start, a, x)` times linear time proportional to
    `n_start`, then each additional term takes another work unit.

    Reference: http://functions.wolfram.com/Polynomials/LaguerreL3/17/01/01/01/
    """
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([laguerre(n_start, a, x)])
    out = np.empty((n_end - n_start, ), dtype=np.float64)
    out[0] = laguerre(n_start, a, x)
    out[1] = laguerre(n_start + 1, a, x)
    for n in range(2, n_end - n_start):
        out[n] = ((a + 2*n - x - 1) * out[n - 1] - (a + n - 1) * out[n - 2]) / n
    return out

def relative_rabi(lamb_dicke: float, n1: int, n2: int) -> float:
    """
    Get the relative Rabi frequency of a transition coupling motional levels
    `n1` and `n2` with a given Lamb--Dicke parameter.  The actual Rabi frequency
    will be the return value multiplied by the base Rabi frequency.
    """
    ldsq = lamb_dicke * lamb_dicke
    out = np.exp(-0.5 * ldsq) * (lamb_dicke ** abs(n1 - n2))
    out = out * laguerre(min(n1, n2), abs(n1 - n2), ldsq)
    fact = 1.0
    for n in range(1 + min(n1, n2), 1 + max(n1, n2)):
        fact = fact * n
    return out / np.sqrt(fact)

def relative_rabi_range(lamb_dicke: float, n_start: int, n_end: int, diff: int)\
        -> np.ndarray:
    """
    Get a range of Rabi frequencies in linear time of `n_end`.  The
    calculation of a single Rabi frequency is linear in `n`, so the naive
    version of a range is quadratic.  This method is functionally equivalent
    to
        np.array([rabi(n, n + diff) for n in range(n_start, n_end)])
    but runs in linear time."""
    if diff < 0:
        n_start = n_start + diff
        n_end = n_end + diff
        diff = -diff
    if n_start >= n_end:
        return np.array([])
    elif n_start == n_end - 1:
        return np.array([relative_rabi(lamb_dicke, n_start, n_start + diff)])
    ldsq = lamb_dicke * lamb_dicke
    const = np.exp(-0.5*ldsq) * lamb_dicke**diff
    lag = laguerre_range(n_start, n_end, diff, ldsq)
    fact = np.empty_like(lag)
    # the np.arange must contain a float so that the `.prod()` call doesn't
    # use fixed-length integers and overflow the factorial calculation.
    fact[0] = 1.0 / np.arange(n_start + 1.0, n_start + diff + 1).prod()
    for i in range(1, n_end - n_start):
        fact[i] = fact[i - 1] * (n_start + i) / (n_start + i + diff)
    return const * lag * np.sqrt(fact)

light_speed   = 299792458       # m/s (exact)
hbar          = 1.054571800e-34 # Js (CODATA 2014, error +-13)
electron_mass = 9.10938356e-31  # kg (CODATA 2014, error +-11)
kg_per_u      = 1.66053904e-27  # kg/u (CODATA 2014, error +-2)
ca40_mass     = 39.962590863    # u (NIST Atomic and Molecular Data, error +-22)
ion_mass      = ca40_mass * kg_per_u - electron_mass # ignores binding energy
carrier_frequency = 2 * np.pi * 411.03174e12 # rad THz (Pavel's thesis, B=1.9T)

def lamb_dicke(motional_frequency, angle=0):
    k = carrier_frequency * np.cos(angle) / light_speed
    return k * np.sqrt(hbar / (2 * ion_mass * motional_frequency))
