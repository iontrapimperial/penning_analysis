"""
Definition of the Axial class, which provides the model for fitting to frequency
scans taken of the axial mode.
"""

import numpy as np
from .variables import *
from ..data import independents, probabilities

class Axial:
    """
    Class used for modelling and fitting an axial spectrum.  The typical
    use-case is fitting to find the mean value of `n` for a thermal state after
    Doppler and/or sideband cooling on the axial mode.

    The x-axis of the data is the frequency of the 729 laser (measured in
    angular Hz detuning from the carrier frequency), and the y-axis is the
    probability that 1 ion was excited.

    The main method is `model()`, which calculates the expected excitation
    probability at a given frequency for some fitted/fitable parameters.
    """
    def __init__(self, data_files, max_n, wait_time, sideband_range=2.5):
        """
        Arguments --
        data_files: list or single SpectrumDataFile --
            The loaded data files which will determine the frequency range to
            consider.
        max_n: int --
            The maximum value of the phonon number to sum upto when considering
            excitation probabilities.
        wait_time: float in s --
            The time over which PMT counts were detected.  Also called the
            "detection time".  This is usually in the file name of the hex
            files, which you can see in the `notes` field of any of the
            `SpectrumDataFile`s.
        sideband_range: ?float --
            The maximum distance the centre of a sideband can be to be
            considered in the sum.  For example, if `sideband_range == 2`, then
            we will sum the two sidebands on either side which have their
            centres nearest to the data point under consideration.  The default
            value is 2.5, which should be sensible unless the Lamb--Dicke
            parameter is extraordinarily high.
        """
        try:
            self.files = list(data_files)
        except TypeError:
            self.files = [data_files]
        self.motional = self.files[0].axial
        for file in self.files:
            if abs(file.axial - self.motional) / self.motional > 1e-8:
                raise ValueError("".join([
                    f"Files '{self.files[0].file_name}' and '{file.file_name}'",
                    f" have axial frequencies {self.motional} and {file.axial}",
                    " which are different.",
                    "  I can't make a unified model for them.",
                ]))
        self.wait_time = wait_time
        self.max_n = max_n
        self.sideband_range = abs(sideband_range)
        self.frequencies = np.concatenate([independents(file)
                                           for file in self.files])
        #self.probabilities =\
        #    np.concatenate([probabilities(file, 8, 3)[1]['probability']
        #                    for file in self.files])
        order = self.frequencies.argsort()
        self.frequencies = self.frequencies[order]
        #self.probabilities = self.probabilities[order]
        min_ = np.ceil(self.frequencies[0]/self.motional - self.sideband_range)
        max_ = np.floor(self.frequencies[-1]/self.motional -self.sideband_range)
        upper = int(max(abs(min_), abs(max_)))
        self.upper = upper

    def __update(self, mean_n, base_rabi, motional_shift):
        """
        Update the pre-calculated quantities ready for being used in the
        summation.  Does not return anything.
        """
        self.lamb_dicke = lamb_dicke(self.motional)
        self.rabi_sq = np.array([
            relative_rabi_range(self.lamb_dicke, 0, self.max_n + 1, s)
            for s in range(self.upper + 1)])
        self.rabi_sq = (base_rabi * base_rabi) * (self.rabi_sq ** 2)
        thermal_ratio = mean_n / (mean_n + 1)
        # Scaling factor to account for loss of unity probabaility by
        # truncation.  Prevent fitting from using negative mean_n or high enough
        # mean_n such that there is >~5% truncation error.
        scale = 1.0 / (1.0 - thermal_ratio**(self.max_n + 1))
        if mean_n < 0 or scale > 1.05:
            self.thermal = np.full(self.max_n + 1, np.inf)
        else:
            # vectorised computation of scale * (nbar/(nbar+1))**n / (nbar+1).
            self.thermal =\
                np.geomspace(scale / (mean_n + 1),
                             thermal_ratio ** self.max_n * scale / (mean_n + 1),
                             self.max_n + 1)

    def __considered_sidebands(self, from_carrier: float) -> np.array:
        """
        Return an array of integers which represent the sidebands that should be
        used for this detuning frequency.  `from_carrier` is a detuning from the
        carrier as an angular frequency in Hz.
        """
        # Use self.motional rather than with the shift so we can't overrun the
        # bounds of the precalculated rabis if the fit goes wrong.  The
        # difference will be a nearly unmeasurable effect on edge-cases anyway.
        min = int(np.ceil(from_carrier / self.motional - self.sideband_range))
        max = int(np.floor(from_carrier / self.motional + self.sideband_range))
        min = min if abs(min) <= self.upper else np.sign(min) * self.upper
        max = max if abs(max) <= self.upper else np.sign(max) * self.upper
        return np.arange(min, max + 1, dtype=np.int32)

    def __excitation(self, sideband, detuning):
        """
        The probability of excitation of the ion due to one sideband but using
        all the considered motional states, at this detuning of the 729 laser.
        """
        rabi_mod_sq = detuning * detuning + self.rabi_sq[abs(sideband)]
        sin_sq = np.sin(0.5 * self.wait_time * np.sqrt(rabi_mod_sq)) ** 2
        n_start = abs(min(sideband, 0))
        n_end = self.max_n + 1 - n_start
        return np.sum(self.thermal[n_start:] * sin_sq[:n_end]\
                      * self.rabi_sq[abs(sideband),:n_end]\
                      / rabi_mod_sq[:n_end])

    def __model_single(self, w, mean_n, base_rabi, carrier_shift,
                       motional_shift, update=True):
        """
        Finds the expected frequency for a single angular frequency `w` with the
        given parameters.  The frequency is given as a detuning from the
        carrier, ignoring the effects of `carrier_shift`.
        """
        if update:
            self.__update(mean_n, base_rabi, motional_shift)
        from_carrier = w + carrier_shift
        motional = self.motional + motional_shift
        total = 0.0
        for sideband in self.__considered_sidebands(from_carrier):
            total += self.__excitation(sideband,
                                       from_carrier - sideband * motional)
        return total

    def __model_many(self, ws, mean_n, base_rabi, carrier_shift,
                     motional_shift):
        """
        Finds the expected frequency for many angular frequencies `ws` with the
        given parameters.  The frequency is given as a detuning from the
        carrier, ignoring the effects of `carrier_shift`.

        This function is largely just a wrapper around `__model_single()`, but
        ensures that the things that need to be updated are only calculated
        once, rather than many times.
        """
        self.__update(mean_n, base_rabi, motional_shift)
        return np.array([self.__model_single(w, mean_n, base_rabi,
                                             carrier_shift, motional_shift,
                                             update=False)
                         for w in ws])

    def model(self, ws, mean_n, base_rabi, carrier_shift, motional_shift,
              offset=0):
        """
        Calculate the probability of excitation for an ion in a thermal state,
        when the 729 laser is detuning by a certain amount from the carrier
        transition.

        Arguments --
        ws: float or np.array of float in angular Hz --
            The frequency or frequencies of the 729 laser, given as a detuning
            from the expected value of the carrier, measured in angular Hz.
            The expected value of the carrier is 411.03174 * 2pi THz.

        mean_n: float > 0 --
            The mean value of motional excitation, assuming that the ion is in a
            thermal state.

        base_rabi: float in angular Hz > 0 --
            The value of the Rabi frequency of the carrier, if the Lamb--Dicke
            parameter could be set to 0.  This is often written `\\Omega_0` in
            the theses, and is the scaling factor for all the sideband and
            state-specific Rabi frequencies.

        carrier_offset: float in angular Hz --
            The actual centre point of the carrier peak, if this differs from
            the 0 point that the frequencies are measured relative to.  This
            ought to be much smaller than the axial frequency.

        motional_offset: float in angular Hz --
            The difference of the actual axial frequency from the value given in
            the data files.

        offset: float > 0 --
            A vertical offset for the baseline of the probability distribution.
            If not given, this just defaults to 0 (so you can omit the quantity
            from the fit by just not passing it).  This physically is a kind of
            measure of the "false negative" rate.
        """
        model = self.__model_many if hasattr(ws, "__len__")\
                else self.__model_single
        prob = model(ws, mean_n, base_rabi, carrier_shift, motional_shift)
        return offset + (1 - offset) * prob
