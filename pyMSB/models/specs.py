from dataclasses import field

from pyMSB.units_conversions import channel_span_to_velocity_span, velocity_span_to_channel_span

from .core import (
    AnalysisGeneric,
    BaseVariable,
    CalibrationGeneric,
    DoubletGeneric,
    LineWidthCoupling,
    SextetGeneric,
    SingletGeneric,
    SpectroscopeGeneric,
    SpectrumGeneric,
)


class SpecsVar(BaseVariable):
    """
    Specification Variable.

    Represents a user estimate of a parameter. Primary use is as a input to fitting models.

    Attributes
    ----------
    value : ``float`` or ``int``
        Value of the fit parameter
    is_fixed : ``bool``
        Flag indicating if the parameter is fixed during fitting

    """

    value: float | int  # | Variable  # | np.ndarray |TensorVariable
    is_fixed: bool = False

    def __init__(self, value: float | int, *, is_fixed: bool = False):
        """Override the constructor to allow first argument `value` to be passed as a positional argument."""
        super().__init__(value=value, is_fixed=is_fixed)

    @classmethod
    def _init_type_mapping(cls):
        cls._type_mapping = {
            SingletGeneric: SingletSpecs,
            DoubletGeneric: DoubletSpecs,
            SextetGeneric: SextetSpecs,
            SpectrumGeneric: SpectrumSpecs,
            SpectroscopeGeneric: SpectroscopeSpecs,
            CalibrationGeneric: CalibrationSpecs,
            AnalysisGeneric: AnalysisSpecs,
        }

    def channel_span_to_velocity_span(self, scale: float) -> "SpecsVar":
        return self.__class__(value=channel_span_to_velocity_span(self.value, scale), is_fixed=self.is_fixed)

    def velocity_span_to_channel_span(self, scale: float) -> "SpecsVar":
        return self.__class__(value=velocity_span_to_channel_span(self.value, scale), is_fixed=self.is_fixed)


class SingletSpecs(SingletGeneric[SpecsVar]):
    """
    Singlet subspectrum with attributes as specification variables (user estimate of singlet subspectrum).

    Primary use is as input to fitting models.

    Attributes
    ----------
    amplitude : :any:`SpecsVar`
        Amplitude of the singlet
    isomer_shift : :any:`SpecsVar`
        Isomer shift of the singlet
    line_width1 : :any:`SpecsVar`
        Line width of the singlet

    """

    pass


class DoubletSpecs(DoubletGeneric[SpecsVar]):
    """
    Doublet subspectrum with attributes as specification variables (user estimate of doublet subspectrum).

    Primary use is as input to fitting models.

    Attributes
    ----------
    amplitude : :any:`SpecsVar`
        Amplitude of the doublet
    isomer_shift : :any:`SpecsVar`
        Isomer shift of the doublet
    quadrupole_split : :any:`SpecsVar`
        Quadrupole split of the doublet
    line_width1 : :any:`SpecsVar`
        Line width of the first line
    line_width2 : :any:`SpecsVar`
        Line width of the second line
    line_width_coupling : :any:`LineWidthCoupling`
        Line width coupling option

    """

    line_width_coupling: LineWidthCoupling = LineWidthCoupling.UNCOUPLED  # is not optional


class SextetSpecs(SextetGeneric[SpecsVar]):
    """
    Sextet subspectrum with attributes as specification variables (user estimate of sextet subspectrum).

    Primary use is as input to fitting models.

    Attributes
    ----------
    amplitude : :any:`SpecsVar`
        Amplitude of the sextet
    isomer_shift : :any:`SpecsVar`
        Isomer shift of the sextet
    quadrupole_split : :any:`SpecsVar`
        Quadrupole split of the sextet
    ratio13 : :any:`SpecsVar`
        Ratio of the amplitudes of the first and third lines
    ratio23 : :any:`SpecsVar`
        Ratio of the amplitudes of the second and third lines
    magnetic_field : :any:`SpecsVar`
        Magnetic field of the sextet
    line_width1 : :any:`SpecsVar`
        Line width of the first line
    line_width2 : :any:`SpecsVar`
        Line width of the second line
    line_width3 : :any:`SpecsVar`
        Line width of the third line
    line_width4 : :any:`SpecsVar`
        Line width of the fourth line
    line_width5 : :any:`SpecsVar`
        Line width of the fifth line
    line_width6 : :any:`SpecsVar`
        Line width of the sixth line
    line_width_coupling : :any:`LineWidthCoupling`
        Line width coupling option

    """

    line_width_coupling: LineWidthCoupling = LineWidthCoupling.UNCOUPLED  # is not optional


class SpectrumSpecs(SpectrumGeneric[SpecsVar]):
    """
    Mossbauer spectrum with attributes as specification variables (user estimate of Mossbauer spectrum).

    Primary use is as input to fitting models.

    Attributes
    ----------
    background : :any:`SpecsVar`
        Background of the spectrum
    singlets : ``list`` [:any:`SingletSpecs`]
        List of singlets
    doublets : ``list`` [:any:`SingletSpecs`]
        List of doublets
    sextets : ``list`` [:any:`SingletSpecs`]
        List of sextets

    """

    singlets: list[SingletSpecs] = field(default_factory=list)
    doublets: list[DoubletSpecs] = field(default_factory=list)
    sextets: list[SextetSpecs] = field(default_factory=list)


class SpectroscopeSpecs(SpectroscopeGeneric[SpecsVar]):
    """
    Mossbauer spectroscope with attributes as specification variables (user estimate of Mossbauer spectroscope).

    Primary use is as input to calibration fitting models.

    Attributes
    ----------
    scale : :any:`SpecsVar`
        Scale of the spectroscope (channels per mm/s)
    isomer_shift_ref : :any:`SpecsVar`
        Reference isomer shift of the spectroscope (position of the energetic zero in channels)

    """

    pass


class CalibrationSpecs(CalibrationGeneric[SpecsVar]):
    """
    Mossbauer calibration with attributes as specification variables (user estimate of Mossbauer calibration).

    Primary use is as input to calibration fitting models.

    Attributes
    ----------
    spectrum : :any:`SpectrumSpecs`
        Mossbauer spectrum
    scope : :any:`SpectroscopeSpecs`
        Mossbauer spectroscope

    """

    pass


class AnalysisSpecs(AnalysisGeneric[SpecsVar]):
    """
    User estimate for Mossbauer analysis.

    Mossbauer analysis with the spectrum as specification variables. The scope is always :any:`SpectroscopeComputable`.
    Primary use is as input to fitting models.

    Attributes
    ----------
    spectrum : :any:`SpectrumSpecs`
        Mossbauer spectrum
    scope : :any:`SpectroscopeComputable`
        Mossbauer spectroscope

    """

    pass
