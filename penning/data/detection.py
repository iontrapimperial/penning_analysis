"""
Tools and methods for detection of fluourescence in the ion trap. The
distributions are available as `dark_pmf()` and `light_pmf()`,
and the measurement estimators are available as `threshold_estimator()` and
`maximum_likelihood_estimator()`.  The biased method of thresholding (i.e. the
naive count of how many points are below a threshold) is available as
`biased_threshold_estimator()`.
"""

import warnings
import numpy as np
from scipy.special import gammaincc, erfc, erf, xlogy

def _integrate(dx, f):
    """
    Numerically integrate a function which is closed on both ends.
    """
    coefficients = np.ones(f.size, dtype=np.float64)
    coefficients[:3] = [3/8, 7/6, 23/24]
    coefficients[-3:] = [23/24, 7/6, 3/8]
    return dx * coefficients.dot(f)

def poisson_pmf(mean, ns):
    """
    Calculate the probability mass function of the Poisson distribution of a
    given mean for all the possible values of `n` up to `ns - 1` (i.e. the
    output array has size `ns`).

    Arguments --
    mean: float >= 0 -- The mean of the distribution (also called lambda).
    ns: int > 0 -- The number of `ns` to calculate the distribution for.

    Returns --
    np.array(dtype=np.float64, shape=(ns,)) --
        The values of the PMF, such that `poisson_pmf(mean, ns)[n] == pmf[n]`.
    """
    out = np.empty(ns, dtype=np.float64)
    out[0] = np.exp(-mean)
    for n in range(1, ns):
        out[n] = out[n - 1] * (mean / n)
    return out

def dark_pmf(dark_rate, detect_time, ns):
    """
    Calculate the probability mass function of the distribution of the count
    rates of a dark ion.

    This is just a Poisson distribution of mean `dark_rate * detect_time`.

    Arguments --
    dark_rate: float in Hz >= 0 -- The photon emission rate for the dark state.
    detect_time: float in s >= 0 -- The time to detect for.
    ns: int > 0 -- The number of `ns` to calculate the distribution for.

    Returns --
    np.array(dtype=np.float64, shape=(ns,)) --
        The values of the PMF, such that `dark_pmf(mean, ns)[n] == pmf[n]`.
    """
    return poisson_pmf(dark_rate * detect_time, ns)

def light_pmf(dark_rate, light_rate, j_mix_decay, detect_time, ns):
    """
    Calculate the probability mass function of the distribution of the count
    rates of a light ion.

    This is a combination of a Poisson distribution, and the j-mixing tail.

    Arguments --
    dark_rate: float in Hz >= 0 -- The photon emission rate of the dark state.
    light_rate: float in Hz >= 0 -- The photon emission rate of the light state.
    j_mix_decay: float in s >= 0 -- The time constant of the j-mixing decay.
    detect_time: float in s >= 0 -- The time to detect for.
    ns: int > 0 -- The number of `ns` to calculate the distribution for.

    Returns --
    np.array(dtype=np.float64, shape=(ns,)) --
        The values of the PMF, such that `light_pmf(mean, ns)[n] == pmf[n]`.
    """
    z_scale = 1 + 1 / (light_rate * j_mix_decay)
    dark_z = dark_rate * detect_time * z_scale
    light_z = (light_rate + dark_rate) * detect_time * z_scale
    recip_r = light_rate * j_mix_decay / (light_rate * j_mix_decay + 1)
    ratio = recip_r * np.exp(dark_rate*detect_time / (light_rate*j_mix_decay))\
            / (light_rate * j_mix_decay)
    j_mix = np.empty(ns, dtype=np.float64)
    for n in range(ns):
        j_mix[n] = ratio * (gammaincc(n+1, dark_z) - gammaincc(n+1, light_z))
        ratio *= recip_r
    poisson = poisson_pmf((dark_rate+light_rate) * detect_time, ns)\
              * np.exp(-detect_time / j_mix_decay)
    return j_mix + poisson

def _index_ceil(needle, haystack, min=0, max=None):
    """
    Find the first index in the `haystack` whose element is greater than or
    equal to `needle`.  This saturates at the top, so if `needle` is greater
    than all the values in the `haystack`, then the returned index is
    `haystack.size - 1`.
    """
    max = max if max is not None else haystack.shape[0] - 1
    mid = min + (max - min) // 2
    if mid == 0:
        # compare to 0 first to avoid accessing haystack[-1]
        return int(haystack[mid] < needle)
    elif mid == haystack.shape[0] - 2 and haystack[mid] < needle:
        # saturate at top - assume needle always lower than max value in
        # haystack.  This might not be the case, but is the desired behaviour in
        # the case of truncation errors in the statistical distributions.
        return mid + 1
    elif haystack[mid] >= needle:
        if haystack[mid - 1] < needle:
            return mid
        else:
            return _index_ceil(needle, haystack, min=min, max=mid)
    else:
        return _index_ceil(needle, haystack, min=mid, max=max)

def sample(distribution, n_samples=1):
    cum_dist = np.cumsum(distribution)
    uniform = np.random.random_sample(n_samples)
    return np.array([_index_ceil(p, cum_dist) for p in uniform],
                     dtype=np.int32)

def histogram(samples, n_max=None, scale=False):
    if n_max is None:
        n_max = int(np.max(samples))
    out = np.zeros(n_max + 1, dtype=np.int32)
    for s in samples:
        out[int(s)] += 1
    return out / samples.shape[0] if scale else out

def log_likelihood(counts, dark, light):
    light_zeros = light == 0
    dark_zeros = dark == 0
    all_light_impossible = np.any(counts[light_zeros])
    all_dark_impossible = np.any(counts[dark_zeros])
    def f(p):
        # we could have precalculated `diff = dark - light` so that this line is
        # `p * diff + light`, which is one fewer FLOP, but this lends itself to
        # greater FP errors when `p ~ 1`.
        probs = p * dark + (1 - p) * light
        fail = (p == 0 and all_light_impossible)\
               or (p == 1 and all_dark_impossible)\
               or np.any(counts[probs == 0])
        return np.sum(xlogy(counts, probs)) if not fail else -np.inf
    return f

def d_log_likelihood_0(counts, dark, light):
    if np.any(np.logical_and(counts, np.logical_not(light))):
        return np.inf
    mask = np.logical_or(counts, light)
    safes = np.sum(counts[mask] * dark[mask] / light[mask])
    unsafes = np.sum(dark[np.logical_not(mask)])
    return (safes + unsafes) / np.sum(counts) - 1

def d_log_likelihood_1(counts, dark, light):
    if np.any(np.logical_and(counts, np.logical_not(dark))):
        return -np.inf
    mask = np.logical_or(counts, dark)
    safes = np.sum(counts[mask] * light[mask] / dark[mask])
    unsafes = np.sum(light[np.logical_not(mask)])
    return 1 - (safes + unsafes) / np.sum(counts)

def _prepare_distributions(dark, light):
    dark = dark if abs(np.sum(dark) - 1) < 1e-8 else dark / np.sum(dark)
    light = light if abs(np.sum(light) - 1) < 1e-8 else light / np.sum(light)
    d_size = 1 + max((np.nonzero(dark)[0][-1], np.nonzero(light)[0][-1]))
    dark, light = [np.pad(x, ((0, max(d_size - x.size, 0)),),
                          'constant', constant_values=0)
                   for x in [dark, light]]
    return dark[:d_size], light[:d_size]

def _crop_counts(counts, dark, light):
    out_of_range = np.sum(counts[dark.size:])
    if out_of_range != 0:
        wmsg = f"{out_of_range} measurements are outside the range"\
               + " of probabilities for the distributions.  These points will"\
               + " be omitted in calculations."
        warnings.warn(wmsg)
    size = min((dark.size, counts.size))
    return counts[:size], dark[:size], light[:size]

# This is actually the golden ratio minus 1, but that's what's actually
# important for the section search (and the ratio of the two sections is the
# golden ratio).
_golden_ratio = 0.5 * (np.sqrt(5) - 1)
def _bracket_maximum(f):
    """
    Initially bracket the maximum, keeping the ratio of points equal.  Only one
    of the `while` branches should be actually enterable, but while the `while`s
    operate like `if`s, there's no semantic `elwhile` statement.
    """
    far, pivot, near = f(0.0), f(_golden_ratio), f(1.0)
    while pivot < far:
        near = pivot
        pivot = f((1 - _golden_ratio) * far[1] + _golden_ratio * pivot[1])
    while pivot < near:
        far = pivot
        pivot = f((1 - _golden_ratio) * pivot[1] + _golden_ratio * near[1])
    return far, pivot, near

def maximum_likelihood_estimator(dark, light, atol=1e-8):
    dark_0, light_0 = _prepare_distributions(dark, light)
    def p(counts):
        counts, dark, light = _crop_counts(counts, dark_0, light_0)
        f_inner = log_likelihood(counts, dark, light)
        f = lambda p: (f_inner(p), p)
        # clip to the relevant edge if the derivatives point away from [0, 1]
        if d_log_likelihood_0(counts, dark, light) <= 0.0:
            return 0.0
        elif d_log_likelihood_1(counts, dark, light) >= 0.0:
            return 1.0
        far, pivot, near = _bracket_maximum(f)
        while abs(near[1] - far[1]) > atol:
            new = f(far[1] + _golden_ratio * (pivot[1] - far[1]))
            if new > pivot:
                far, pivot, near = far, new, pivot
            else:
                far, pivot, near = near, pivot, new
        return pivot[1]
    return p

def maximum_likelihood_std(dark, light):
    dark, light = _prepare_distributions(dark, light)
    diff_sq = (dark - light) ** 2
    diff_sq_zeros = diff_sq == 0
    mask = np.logical_not(diff_sq_zeros)
    def std(p, n):
        scalar = np.isscalar(p) and np.isscalar(n)
        p, n = (np.array([p]), np.array([n])) if scalar\
               else np.broadcast_arrays(p, n)
        # We define `0/0 == 0` here so that including additional bins beyond the
        # necessary ones does not change the Fisher information.
        var = np.zeros(p.shape, dtype=np.float64)
        bin_p = np.outer(p, dark) + np.outer(1 - p, light)
        out = np.all(np.logical_or(diff_sq_zeros, bin_p), axis=1)
        ps, ns = p[out], n[out]

        # Scaled variance and standard deviation of the unclipped estimators.
        # Except in one case, the quantities are always used as `2 * var` or
        # `sqrt(2) * std`, so this saves us a few vector operations.
        ma_bin_p = bin_p[np.outer(out, mask)].reshape(np.sum(out), np.sum(mask))
        fisher = np.sum(diff_sq[mask] / ma_bin_p, axis=1)
        var_p = 2 / (ns * fisher)
        std_p = np.sqrt(var_p)

        # Precalculate a whole bunch of quantities to save time.
        _1_p = 1 - ps
        g_1_p_sq = np.exp(- _1_p**2 / var_p)
        g_p_sq = np.exp(- ps**2 / var_p)
        recip_std_p = 1.0 / std_p
        erfc_1_p = erfc(_1_p * recip_std_p)
        erf_1_p = erf(_1_p * recip_std_p)
        erf_p = erf(ps * recip_std_p)
        recip_sqrt_pi = 1.0 / np.sqrt(np.pi)

        # Expectation of the unclipped estimator (perhaps unnecessary).
        exp_cp = 0.5 * ((g_p_sq - g_1_p_sq) * std_p * recip_sqrt_pi
                        + ps * (erf_1_p + erf_p) + erfc_1_p)

        exp_p_sq = 0.5 * ((ps**2 + 0.5*var_p) * (erf_1_p + erf_p)
                          + std_p * recip_sqrt_pi * (ps * g_p_sq
                                                     - (ps + 1) * g_1_p_sq)
                          + erfc_1_p)
        var[out] = exp_p_sq - exp_cp**2
        return np.sqrt(var) if not scalar else np.sqrt(var[0])
    return std

def biased_threshold_estimator(threshold):
    return lambda counts: np.sum(counts[:threshold]) / np.sum(counts)

def threshold_integral_std(dark, light, t):
    d = np.sum(dark[t:])
    l = np.sum(light[:t])
    scale = (1 - d - l)**2
    if scale < 1e-10:
        return np.inf
    scale = 0.125 / scale
    return scale * (2 * ((1 - 2*d) * np.sqrt(d*(1-d))
                         + (1 - 2*l) * np.sqrt(l*(1-l)))
                    + np.arcsin(1 - 2*l) + np.arcsin(1 - 2*d))

def best_threshold(dark, light):
    return np.argmin([threshold_integral_std(dark, light, t)
                      for t in np.arange(min((dark.size, light.size)))])

def threshold_std(dark, light, t=None):
    if t is None:
        t = best_threshold(dark, light)
    l = np.sum(light[:t])
    fidelity = 1 - l - np.sum(dark[t:])
    if fidelity < 1e-8:
        # catch negative case too.
        return lambda p, n: np.full(np.broadcast(p, n).shape, np.inf)
    scale = 1.0 / fidelity
    c0, c1, c2 = [l * (1 - l),
                  fidelity * (1 - 2 * l),
                  -fidelity * fidelity]
    return lambda p, n: np.sqrt((c0 + p * (c1 + p * c2)) / (n - 1)) * scale

def threshold_estimator(dark, light):
    t = best_threshold(dark, light)
    l_error = np.sum(light[:t])
    scale = 1.0 / (1 - l_error - np.sum(dark[t:]))
    return lambda counts: (np.sum(counts[:t])/np.sum(counts) - l_error) * scale
