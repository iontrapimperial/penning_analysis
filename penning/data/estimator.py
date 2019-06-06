import abc
import warnings
import numpy as np
import scipy.special

def _prepare_distributions(dark, light):
    """
    Return two distributions cropped to be the same size, and exactly large
    enough to hold all of the non-zero values from before.
    """
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
               + " be omitted in calculations.  This might imply that the"\
               + " distributions were not measured correctly."
        warnings.warn(wmsg)
    size = min((dark.size, counts.size))
    return counts[:size], dark[:size], light[:size]

class Estimator(abc.ABC):
    @abc.abstractmethod
    def estimate(self, counts):
        """
        Calculate an estimator of the probability of excitation and its
        uncertainty from a set of PMT counts.

        Arguments --
        counts: array_like of int > 0 --
            A histogram of the number of times each number of counts was
            detected on the PMT.  The array index is the number of counts, and
            the value is the number of times that count rate was seen.

        Returns --
        p: float -- The estimated probability.
        std: float -- The estimated uncertainty in the probability.
        """

class SimpleThreshold(Estimator):
    """
    A naive thresholding estimator.  This must not be used for published data
    because the probability estimator is biased, but is provided for quick and
    dirty visualisation of results without a full probability distribution scan.
    """
    def __init__(self, threshold):
        """
        Arguments --
        threshold: int >= 0 --
            The fixed threshold to use.  All counts lower than this will count
            as detection of the excited state, and all counts equal or higher
            than this will be the ground state.
        """
        self.threshold = threshold

    def estimate(self, counts):
        n = np.sum(counts)
        if n == 0:
            return np.nan, np.inf
        p = np.sum(counts[:self.threshold]) / n
        return p, np.sqrt(p * (1 - p) / n)

class Threshold(Estimator):
    def __init__(self, dark, light, threshold=None):
        self.dark, self.light = _prepare_distributions(dark, light)
        if threshold is None:
            max_t = min((dark.size, light.size))
            self.threshold = np.argmin([self.integral_uncertainty(t)
                                        for t in range(max_t)])
        else:
            self.threshold = threshold
        self.__d = np.sum(self.dark[self.threshold:])
        self.__l = np.sum(self.light[:self.threshold])
        self.__fidelity = 1 - self.__d - self.__l
        if self.__fidelity < 1e-8:
            raise ValueError("The thresholding error is too great.  Are the"
                             + " distributions the correct way around, and is"
                             + " the threshold chosen properly?")
        self.__scale = 1.0 / self.__fidelity
        self.__c = [self.__l * (1 - self.__l),
                    self.__fidelity * (1 - 2 * self.__l),
                    -self.__fidelity * self.__fidelity]

    def p(self, counts):
        return (np.sum(counts[:self.threshold])/np.sum(counts) - self.__l)\
               * self.__scale

    def estimate(self, counts):
        n = np.sum(counts)
        p = (np.sum(counts[:self.threshold])/n - self.__l) * self.__scale
        poly = self.__c[0] + p * (self.__c[1] + p * self.__c[2])
        return p, np.sqrt(poly / n) * self.__scale

    def integral_uncertainty(self, threshold=None):
        if threshold is None:
            d = self.__d
            l = self.__l
        else:
            d = np.sum(self.dark[threshold:])
            l = np.sum(self.light[:threshold])
        fidelity = (1 - d - l)**2
        if fidelity < 1e-10:
            return np.inf
        scale = 0.125 / fidelity
        return scale * (2 * ((1 - 2 * d) * np.sqrt(d * (1 - d))
                             + (1 - 2 * l) * np.sqrt(l * (1 - l)))
                        + np.arcsin(1 - 2 * d) + np.arcsin(1 - 2 * l))

class MLE(Estimator):
    def __init__(self, dark, light):
        self.dark, self.light = _prepare_distributions(dark, light)
        self.__p_limits = self.light / (self.light - self.dark)
        self.__diff_sq = (self.dark - self.light) ** 2
        self.__diff_sq_zeros = self.__diff_sq == 0
        self.__mask = np.logical_not(self.__diff_sq_zeros)

    def __d_log_likelihood(self, counts):
        counts, dark, light = _crop_counts(counts, self.dark, self.light)
        counts = counts / np.sum(counts)
        diff = dark - light
        def f(p):
            try:
                binomial_p = p * dark + (1 - p) * light
            except FloatingPointError:
                print(p)
                print(dark)
                print(light)
                raise
            if not np.all(binomial_p):
                return np.inf if p < 0.5 else -np.inf
            return np.sum(counts * diff / binomial_p)
        return lambda p: (p, f(p))

    def __bracket_crossing(self, counts):
        nonzeros = np.nonzero(counts)
        light_cut = self.light[nonzeros]
        dark_cut = self.dark[nonzeros]
        denom = light_cut - dark_cut
        denom_nonzero = denom != 0.0
        p_asymptotes = light_cut[denom_nonzero] / denom[denom_nonzero]
        lower, upper = -0.5,1.0
        for asymptote in p_asymptotes:
            if lower < asymptote <= 0.0:
                lower = asymptote
            elif 1.0 <= asymptote < upper:
                upper = asymptote
        return (lower, np.inf), (upper, -np.inf)

    def __estimate(self, counts, atol=1e-8):
        gradient = self.__d_log_likelihood(counts)
        left, right = self.__bracket_crossing(counts)
        while abs(left[0] - right[0]) > atol:
            mid = gradient(0.5 * (left[0] + right[0]))
            if mid[1] == 0.0:
                return mid[0]
            if mid[1] > 0:
                left = mid
            else:
                right = mid
        return 0.5 * (left[0] + right[0])

    def __std(self, p, n, atol=1e-8):
        binomial_p = p * self.dark + (1 - p) * self.light
        if not np.all(np.logical_or(self.__diff_sq_zeros, binomial_p)):
            return 0
        fisher = np.sum(self.__diff_sq[self.__mask] / binomial_p[self.__mask])
        if fisher < atol:
            return np.inf
        return np.sqrt(1 / (n * fisher))

    def estimate(self, counts):
        p = self.__estimate(counts)
        return p, self.__std(p, np.sum(counts))

_recip_sqrt_pi = 0.5641895835477563
_sqrt_2        = 1.4142135623730951
class Clipped(Estimator):
    """
    Clip an existing estimator to only return probabilities between 0 and 1, and
    update the variance accordingly.
    """
    def __init__(self, estimator: Estimator):
        self.__estimate = estimator.estimate

    def estimate(self, counts):
        p, std = self.__estimate(counts)
        sqrt_2_std = _sqrt_2 * std
        recip_sqrt_2_std = 1.0 / sqrt_2_std
        recip_2_var = recip_sqrt_2_std * recip_sqrt_2_std
        gauss_1_p_sq = np.exp((p-1) * (1-p) * recip_2_var)
        gauss_p_sq = np.exp(-p * p * recip_2_var)
        erfc_1_p = scipy.special.erfc((1 - p) * recip_sqrt_2_std)
        erf_1_p = scipy.special.erf((1 - p) * recip_sqrt_2_std)
        erf_p = scipy.special.erf(p * recip_sqrt_2_std)
        exp_p = 0.5 * ((gauss_p_sq - gauss_1_p_sq) * sqrt_2_std * _recip_sqrt_pi
                       + p * (erf_1_p + erf_p) + erfc_1_p)
        exp_p_sq = 0.5 * ((p * p + std * std) * (erf_1_p + erf_p)
                          + sqrt_2_std * _recip_sqrt_pi
                                * (p * gauss_p_sq - (p + 1) * gauss_1_p_sq)
                          + erfc_1_p)
        p = 0 if p < 0 else (1 if p > 1 else p)
        return p, np.sqrt(exp_p_sq - exp_p * exp_p)
