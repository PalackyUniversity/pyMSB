from typing import Optional

import numpy as np
from pydantic import ConfigDict, field_serializer

from pyMSB.units_conversions import channel_span_to_velocity_span, velocity_span_to_channel_span

from .core import (
    AnalysisGeneric,
    BaseVariable,
    CalibrationGeneric,
    ComputableVar,
    DoubletGeneric,
    SextetGeneric,
    SingletGeneric,
    SpectroscopeGeneric,
    SpectrumGeneric,
)


# Point estimation models
class PointVar(BaseVariable):
    """
    Point result of a parameter estimation.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    value : :any:`ComputableVar`
        Point estimate of the parameter
    sigma : :any:`ComputableVar` or ``None``
        Uncertainty (standard deviation) of the point estimate (optional)

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: ComputableVar
    sigma: Optional[ComputableVar] = None

    def __init__(self, value: ComputableVar, *, sigma: ComputableVar | None = None):
        """Override the constructor to allow first argument `value` to be passed as a positional argument."""
        super().__init__(value=value, sigma=sigma)

    @classmethod
    def _init_type_mapping(cls):
        cls._type_mapping = {
            SingletGeneric: SingletPoint,
            DoubletGeneric: DoubletPoint,
            SextetGeneric: SextetPoint,
            SpectrumGeneric: SpectrumPoint,
            SpectroscopeGeneric: SpectroscopePoint,
            CalibrationGeneric: CalibrationPoint,
        }

    @field_serializer("value", "sigma")
    def serialize_numpy(self, value: ComputableVar):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def channel_span_to_velocity_span(self, scale: float) -> "PointVar":
        return self.__class__(
            value=channel_span_to_velocity_span(self.value, scale),
            sigma=channel_span_to_velocity_span(self.sigma, scale),
        )

    def velocity_span_to_channel_span(self, scale: float) -> "PointVar":
        return self.__class__(
            value=velocity_span_to_channel_span(self.value, scale),
            sigma=velocity_span_to_channel_span(self.sigma, scale),
        )


class SingletPoint(SingletGeneric[PointVar]):
    """
    Singlet subspectrum with attributes as point estimates.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    amplitude : :any:`PointVar`
        Point estimate of the amplitude of the singlet
    isomer_shift : :any:`PointVar`
        Point estimate of the isomer shift of the singlet
    line_width1 : :any:`PointVar`
        Point estimate of the line width of the singlet

    """

    pass


class DoubletPoint(DoubletGeneric[PointVar]):
    """
    Doublet subspectrum with attributes as point estimates.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    amplitude : :any:`PointVar`
        Point estimate of the amplitude of the doublet
    isomer_shift : :any:`PointVar`
        Point estimate of the isomer shift of the doublet
    quadrupole_split : :any:`PointVar`
        Point estimate of the quadrupole split of the doublet
    line_width1 : :any:`PointVar`
        Point estimate of the line width of the first line
    line_width2 : :any:`PointVar`
        Point estimate of the line width of the second line

    """

    pass


class SextetPoint(SextetGeneric[PointVar]):
    """
    Sextet subspectrum with attributes as point estimates.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    amplitude : :any:`PointVar`
        Point estimate of the amplitude of the sextet
    isomer_shift : :any:`PointVar`
        Point estimate of the isomer shift of the sextet
    quadrupole_split : :any:`PointVar`
        Point estimate of the quadrupole split of the sextet
    ratio13 : :any:`PointVar`
        Point estimate of the ratio of the amplitudes of the first and third lines
    ratio23 : :any:`PointVar`
        Point estimate of the ratio of the amplitudes of the second and third lines
    magnetic_field : :any:`PointVar`
        Point estimate of the magnetic field of the sextet
    line_width1 : :any:`PointVar`
        Point estimate of the line width of the first line
    line_width2 : :any:`PointVar`
        Point estimate of the line width of the second line
    line_width3 : :any:`PointVar`
        Point estimate of the line width of the third line
    line_width4 : :any:`PointVar`
        Point estimate of the line width of the fourth line
    line_width5 : :any:`PointVar`
        Point estimate of the line width of the fifth line
    line_width6 : :any:`PointVar`
        Point estimate of the line width of the sixth line

    """

    pass


class SpectrumPoint(SpectrumGeneric[PointVar]):
    """
    Mossbauer spectrum with attributes as point estimates.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    background : :any:`PointVar`
        Point estimate of the background of the spectrum
    singlets : ``list`` [:any:`SingletPoint`]
        List of singlets
    doublets : ``list`` [:any:`SingletPoint`]
        List of doublets
    sextets : ``list`` [:any:`SingletPoint`]
        List of sextets

    """

    pass


class SpectroscopePoint(SpectroscopeGeneric[PointVar]):
    """
    Mossbauer spectroscope with attributes as point estimates.

    Primary use is as output of point estimate fitting models (MAP, LSQ).

    Attributes
    ----------
    scale : :any:`PointVar`
        Point estimate of the scale of the spectroscope (channels per mm/s)
    isomer_shift_ref : :any:`PointVar`
        Point estimate of the reference isomer shift of the spectroscope (position of the energetic zero in channels)

    """

    pass


class CalibrationPoint(CalibrationGeneric[PointVar]):
    """
    Mossbauer calibration with attributes as point estimates.

    Primary use is as output of point estimate calibration fitting models (MAP, LSQ).

    Attributes
    ----------
    spectrum : :any:`SpectrumPoint`
        Mossbauer spectrum
    scope : :any:`SpectroscopePoint`
        Mossbauer spectroscope

    """

    pass


class AnalysisPoint(AnalysisGeneric[PointVar]):
    pass
