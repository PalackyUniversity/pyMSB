from typing import Optional

import numpy as np
import xarray
from pydantic import ConfigDict, field_serializer
from pytensor import Variable

from pyMSB.units_conversions import channel_span_to_velocity_span, velocity_span_to_channel_span

from .core import (
    AnalysisGeneric,
    BaseVariable,
    CalibrationGeneric,
    DoubletGeneric,
    SextetGeneric,
    SingletGeneric,
    SpectroscopeGeneric,
    SpectrumGeneric,
)


# Bayesian models
class BayesVar(BaseVariable):
    """
    Full Bayesian result for a parameter estimation.

    Primary use is as output of Bayesian fitting models (FULL_BAYES).

    Attributes
    ----------
    posterior : ``list`` [``float``] or ``np.ndarray``
        Posterior trace of the parameter
    value : ``float``
        Mean of the posterior trace
    sigma : ``float``
        Uncertainty (Standard deviation) of the posterior trace

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    posterior: np.ndarray | xarray.DataArray  # | list[float]

    def posterior_mean(self):
        """Mean of the posterior trace."""
        return np.mean(self.posterior)

    def posterior_std(self):
        """Uncertainty (Standard deviation) of the posterior trace."""
        return np.std(self.posterior)

    @property
    def value(self):
        """Mean of the posterior trace."""
        return self.posterior_mean()

    @property
    def sigma(self):
        """Uncertainty (Standard deviation) of the posterior trace."""
        return self.posterior_std()

    @classmethod
    def _init_type_mapping(cls):
        cls._type_mapping = {
            SingletGeneric: SingletBayes,
            DoubletGeneric: DoubletBayes,
            SextetGeneric: SextetBayes,
            SpectrumGeneric: SpectrumBayes,
            SpectroscopeGeneric: SpectroscopeBayes,
            CalibrationGeneric: CalibrationBayes,
        }

    @field_serializer("posterior")
    def serialize_numpy(self, value: list[float] | np.ndarray):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, xarray.DataArray):
            return value.values.tolist()
        return value

    def channel_span_to_velocity_span(self, scale: float) -> "BayesVar":
        return self.__class__(posterior=channel_span_to_velocity_span(self.posterior, scale))

    def velocity_span_to_channel_span(self, scale: float) -> "BayesVar":
        return self.__class__(posterior=velocity_span_to_channel_span(self.posterior, scale))


class SingletBayes(SingletGeneric[BayesVar]):
    """
    Singlet subspectrum with attributes as Bayesian results.

    Primary use is as output of Bayesian fitting models (FULL_BAYES).

    Attributes
    ----------
    amplitude : :any:`BayesVar`
        Bayesian result of the amplitude of the singlet
    isomer_shift : :any:`BayesVar`
        Bayesian result of the isomer shift of the singlet
    line_width1 : :any:`BayesVar`
        Bayesian result of the line width of the singlet

    """

    pass


class DoubletBayes(DoubletGeneric[BayesVar]):
    """
    Doublet subspectrum with attributes as Bayesian results.

    Primary use is as output of Bayesian fitting models (FULL_BAYES).

    Attributes
    ----------
    amplitude : :any:`BayesVar`
        Bayesian result of the amplitude of the doublet
    isomer_shift : :any:`BayesVar`
        Bayesian result of the isomer shift of the doublet
    quadrupole_split : :any:`BayesVar`
        Bayesian result of the quadrupole split of the doublet
    line_width1 : :any:`BayesVar`
        Bayesian result of the line width of the first line
    line_width2 : :any:`BayesVar`
        Bayesian result of the line width of the second line

    """

    pass


class SextetBayes(SextetGeneric[BayesVar]):
    """
    Sextet subspectrum with attributes as Bayesian results.

    Primary use is as output of Bayesian fitting models (FULL_BAYES).

    Attributes
    ----------
    amplitude : :any:`BayesVar`
        Bayesian result of the amplitude of the sextet
    isomer_shift : :any:`BayesVar`
        Bayesian result of the isomer shift of the sextet
    quadrupole_split : :any:`BayesVar`
        Bayesian result of the quadrupole split of the sextet
    ratio13 : :any:`BayesVar`
        Bayesian result of the ratio of the amplitudes of the first and third lines
    ratio23 : :any:`BayesVar`
        Bayesian result of the ratio of the amplitudes of the second and third lines
    magnetic_field : :any:`BayesVar`
        Bayesian result of the magnetic field of the sextet
    line_width1 : :any:`BayesVar`
        Bayesian result of the line width of the first line
    line_width2 : :any:`BayesVar`
        Bayesian result of the line width of the second line
    line_width3 : :any:`BayesVar`
        Bayesian result of the line width of the third line
    line_width4 : :any:`BayesVar`
        Bayesian result of the line width of the fourth line
    line_width5 : :any:`BayesVar`
        Bayesian result of the line width of the fifth line
    line_width6 : :any:`BayesVar`
        Bayesian result of the line width of the sixth line

    """

    pass


class SpectrumBayes(SpectrumGeneric[BayesVar]):
    """
    Mossbauer spectrum with attributes as Bayesian results.

    Primary use is as output of Bayesian fitting models (FULL_BAYES).

    Attributes
    ----------
    background : :any:`BayesVar`
        Bayesian result of the background of the spectrum
    singlets : ``list`` [:any:`SingletBayes`]
        List of singlets
    doublets : ``list`` [:any:`SingletBayes`]
        List of doublets
    sextets : ``list`` [:any:`SingletBayes`]
        List of sextets

    """

    pass


class SpectroscopeBayes(SpectroscopeGeneric[BayesVar]):
    """
    Mossbauer spectroscope with attributes as Bayesian results.

    Primary use is as output of Bayesian calibration fitting models (FULL_BAYES).

    Attributes
    ----------
    scale : :any:`BayesVar`
        Bayesian result of the scale of the spectroscope (channels per mm/s)
    isomer_shift_ref : :any:`BayesVar`
        Bayesian result of the reference isomer shift of the spectroscope (position of the energetic zero in channels)

    """

    pass


class CalibrationBayes(CalibrationGeneric[BayesVar]):
    """
    Mossbauer calibration with attributes as Bayesian results.

    Primary use is as output of Bayesian calibration fitting models (FULL_BAYES).

    Attributes
    ----------
    spectrum : :any:`SpectrumBayes`
        Mossbauer spectrum result
    scope : :any:`SpectroscopeBayes`
        Mossbauer spectroscope result

    """

    pass


class AnalysisBayes(AnalysisGeneric[BayesVar]):
    """
    Mossbauer analysis with the spectrum as Bayesian results. The scope is always :any:`SpectroscopeComputable`.

    Primary use is as output of Bayesian analysis models (FULL_BAYES).

    Attributes
    ----------
    spectrum : :any:`SpectrumBayes`
        Mossbauer spectrum result
    scope : :any:`SpectroscopeComputable`
        Mossbauer spectroscope

    """

    pass


# Bayesian random variable models
class SpectrumRVs(SpectrumGeneric[Optional[Variable]], validate_assignment=True):
    pass


class SpectroscopeRVs(SpectroscopeGeneric[Optional[Variable]], validate_assignment=True):
    pass


class SingletRVs(SingletGeneric[Optional[Variable]], validate_assignment=True):
    pass


class DoubletRVs(DoubletGeneric[Optional[Variable]], validate_assignment=True):
    pass


class SextetRVs(SextetGeneric[Optional[Variable]], validate_assignment=True):
    pass
