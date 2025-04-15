import abc
from dataclasses import field
from enum import Enum
from typing import Any, Callable, Generic, Self, TypeVar, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pytensor import Variable

from pyMSB.units_conversions import (
    channel_span_to_velocity_span,
    channel_to_velocity,
    velocity_span_to_channel_span,
    velocity_to_channel,
)


class MaterialConstants(BaseModel):
    """
    Material constants for Mossbauer spectroscopy.

    Attributes
    ----------
    zeeman_diff_ground : ``float``
        Material constant - Zeeman split of ground energy state
    zeeman_diff_excited : ``float``
        Material constant - Zeeman split of excited energy state

    .. note::
        Zeeman split is given by the product of nuclear magnetron (0.6561682 mm/s/T) and gyromagnetic ratio, which is\
        different for each chemical element and energy state.

    """

    zeeman_diff_ground: float
    zeeman_diff_excited: float


# ----------------------------------------------------------------------------------------
# Enums
# ----------------------------------------------------------------------------------------


class LineWidthCoupling(Enum):
    """
    Enum for line width coupling options.

    Options
    -------
        - UNCOUPLED - All `line widths` are independent
        - COUPLED_PAIRS - `Line widths` are coupled in pairs (for sextet: 1-6, 2-5, 3-4)
        - COUPLED - All `line widths` are coupled (equal)

    """

    UNCOUPLED = "Uncoupled"  # All parameters are free
    COUPLED_PAIRS = "CoupledPairs"  # Line widths are coupled in pairs
    COUPLED = "Coupled"  # All line widths are coupled


class SpectroscopeGeometry(Enum):
    """
    Enum for spectroscope geometry options.

    Options
    -------
        - TRANSMISSION - Source, sample, and detector are on a line in this order
        - REFLECTION - Source, detector, and sample are on a line in this order

    """

    TRANSMISSION = "Transmission"  # Source, sample, and detector are on a line in this order
    REFLECTION = "Reflection"  # Source, detector, and sample are on a line in this order


# ----------------------------------------------------------------------------------------
# Generic classes
# ----------------------------------------------------------------------------------------


T = TypeVar("T")
T2 = TypeVar("T2")


class BaseVariable(BaseModel, abc.ABC):
    """Base class for all variable types."""

    _type_mapping: dict[type[BaseModel], type[BaseModel]] = PrivateAttr(dict)

    @property
    def value_rounded(self):
        """Value rounded to two significant digits of uncertainty (sigma)."""
        if self.sigma is None or self.sigma == 0:
            return round(self.value, 2)
        if self.sigma < 0:
            raise ValueError(f"Uncertainty {self.sigma} is negative")
        n = int(-np.floor(np.log10(self.sigma / 10)))  # /10 for 2 significant digits of uncertainty
        rounded_value = np.round(self.value, n)
        return rounded_value if n > 0 else int(rounded_value)

    @property
    def sigma_rounded(self):
        """Sigma rounded to two significant digits."""
        if self.sigma is None or self.sigma == 0:
            return self.sigma
        if self.sigma < 0:
            raise ValueError(f"Uncertainty {self.sigma} is negative")
        n = int(-np.floor(np.log10(self.sigma / 10)))  # /10 for 2 significant digits of uncertainty
        rounded_sigma = np.round(self.sigma, n)
        return rounded_sigma if n > 0 else int(rounded_sigma)

    @classmethod
    @abc.abstractmethod
    def _init_type_mapping(cls):
        """Initialize the type mapping. Lazy Initialization is used to avoid circular imports."""
        pass

    @classmethod
    def get_typed_cls(cls, base_cls: type[BaseModel]) -> type[BaseModel]:
        """Get the specific class type corresponding to the variable."""
        # if len(cls._type_mapping) == 0:
        cls._init_type_mapping()  # TODO: Check if this is necessary
        return cls._type_mapping.get(base_cls, base_cls)


def get_typed_cls_from_mapping(mapping: Callable[[T], T2], sample_input: T, base_cls: type[BaseModel]) -> type[T2]:
    output = mapping(sample_input)

    if isinstance(output, ComputableVar):
        return COMPUTABLE_MAPPING.get(base_cls, base_cls)
    elif isinstance(output, BaseVariable):
        return output.get_typed_cls(base_cls)
    raise ValueError(f"Unsupported output type: {type(output)}.")


class BaseMossbauerGeneric(BaseModel, abc.ABC):
    """Base class for all Mossbauer models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def map(self, mapping: Callable[[T], T2]):
        """Map the model using a mapping function."""
        raise NotImplementedError("Method not implemented.")

    @classmethod
    def model_validate(cls, *args, **kwargs) -> Self:
        """
        Override the default method to convert all attributes to the correct type.

        Simple identity mapping is used. Inside the map method, correct type conversion is done.

        Standard model_validate:

        .. code:: python

            cal = CalibrationSpecs.model_validate(data)
            cal.scope: SpectroscopeGeneric[VarSpecs]

        With the override:

        .. code:: python

            cal = CalibrationSpecs.model_validate(data)
            cal.scope: SpectroscopeSpecs

        """
        # Validate JSON data
        model = super().model_validate(*args, **kwargs)
        return model.map(lambda x: x)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


class Subspectrum(BaseMossbauerGeneric, abc.ABC):
    """
    Base class for all subspectra types (Singlet, Doublet, Sextet).

    Attributes
    ----------
    name : ``str`` or ``None``
        Name of the subspectrum (optional)

    """

    # Name is not required and is not used in the mossbauer package right now.
    # Let's see if we find ability to add common subspectra attributes useful in the future.
    name: str | None = field(default=None, kw_only=True)


class SingletGeneric(Subspectrum, Generic[T]):
    """
    Generic class for singlet subspectrum.

    Attributes
    ----------
    amplitude : ``T``
        Amplitude of the singlet
    isomer_shift : ``T``
        Isomer shift of the singlet
    line_width1 : ``T``
        Line width of the singlet

    """

    amplitude: T
    isomer_shift: T
    line_width1: T

    def map(self, mapping: Callable[[T], T2]) -> "SingletGeneric[T2]":
        """
        Map the singlet using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function


        Returns
        -------
        ``SingletGeneric[T2]``
            Mapped singlet - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.amplitude, SingletGeneric)
        return cls(
            name=self.name,
            amplitude=mapping(self.amplitude),
            isomer_shift=mapping(self.isomer_shift),
            line_width1=mapping(self.line_width1),
        )

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "SingletGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SingletGeneric`
            Singlet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.channel_span_to_velocity_span(scope.scale),
            line_width1=self.line_width1.channel_span_to_velocity_span(scope.scale),
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "SingletGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SingletGeneric`
            Singlet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.velocity_span_to_channel_span(scope.scale),
            line_width1=self.line_width1.velocity_span_to_channel_span(scope.scale),
        )


class DoubletGeneric(Subspectrum, Generic[T]):
    """
    Generic class for doublet subspectrum.

    Attributes
    ----------
    amplitude : ``T``
        Amplitude of the doublet
    isomer_shift : ``T``
        Isomer shift of the doublet
    quadrupole_split : ``T``
        Quadrupole split of the doublet
    line_width1 : ``T``
        Line width of the first line
    line_width2 : ``T``
        Line width of the second line

    """

    amplitude: T
    isomer_shift: T
    quadrupole_split: T
    line_width1: T
    line_width2: T
    line_width_coupling: LineWidthCoupling | None = None  # optional

    def map(self, mapping: Callable[[T], T2], **cls_kwargs) -> "DoubletGeneric[T2]":
        """
        Map the doublet using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function
        cls_kwargs : ``dict``
            Additional keyword arguments to be passed to the constructor of the mapped class (used for line width\
            coupling)

        Returns
        -------
        ``DoubletGeneric[T2]``
            Mapped doublet - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.amplitude, DoubletGeneric)
        return cls(
            name=self.name,
            amplitude=mapping(self.amplitude),
            isomer_shift=mapping(self.isomer_shift),
            quadrupole_split=mapping(self.quadrupole_split),
            line_width1=mapping(self.line_width1),
            line_width2=mapping(self.line_width2),
            line_width_coupling=cls_kwargs.get("line_width_coupling", self.line_width_coupling),
        )

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "DoubletGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`DoubletGeneric`
            Doublet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.channel_span_to_velocity_span(scope.scale),
            quadrupole_split=self.quadrupole_split.channel_span_to_velocity_span(scope.scale),
            line_width1=self.line_width1.channel_span_to_velocity_span(scope.scale),
            line_width2=self.line_width2.channel_span_to_velocity_span(scope.scale),
            line_width_coupling=self.line_width_coupling,
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "DoubletGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`DoubletGeneric`
            Doublet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.velocity_span_to_channel_span(scope.scale),
            quadrupole_split=self.quadrupole_split.velocity_span_to_channel_span(scope.scale),
            line_width1=self.line_width1.velocity_span_to_channel_span(scope.scale),
            line_width2=self.line_width2.velocity_span_to_channel_span(scope.scale),
            line_width_coupling=self.line_width_coupling,
        )


class SextetGeneric(Subspectrum, Generic[T]):
    """
    Generic class for sextet subspectrum.

    Attributes
    ----------
    amplitude : ``T``
        Amplitude of the sextet
    isomer_shift : ``T``
        Isomer shift of the sextet
    quadrupole_split : ``T``
        Quadrupole split of the sextet
    ratio13 : ``T``
        Ratio of the amplitudes of the first and third lines
    ratio23 : ``T``
        Ratio of the amplitudes of the second and third lines
    magnetic_field : ``T``
        Magnetic field of the sextet
    line_width1 : ``T``
        Line width of the first line
    line_width2 : ``T``
        Line width of the second line
    line_width3 : ``T``
        Line width of the third line
    line_width4 : ``T``
        Line width of the fourth line
    line_width5 : ``T``
        Line width of the fifth line
    line_width6 : ``T``
        Line width of the sixth line

    """

    amplitude: T
    isomer_shift: T
    quadrupole_split: T
    ratio13: T
    ratio23: T
    magnetic_field: T
    line_width1: T
    line_width2: T
    line_width3: T
    line_width4: T
    line_width5: T
    line_width6: T
    line_width_coupling: LineWidthCoupling | None = None  # optional

    def map(self, mapping: Callable[[T], T2], **cls_kwargs) -> "SextetGeneric[T2]":
        """
        Map the sextet using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function
        cls_kwargs : ``dict``
            Additional keyword arguments to be passed to the constructor of the mapped class (used for line width\
            coupling)

        Returns
        -------
        ``SextetGeneric[T2]``
            Mapped sextet - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.amplitude, SextetGeneric)
        return cls(
            name=self.name,
            amplitude=mapping(self.amplitude),
            isomer_shift=mapping(self.isomer_shift),
            quadrupole_split=mapping(self.quadrupole_split),
            ratio13=mapping(self.ratio13),
            ratio23=mapping(self.ratio23),
            magnetic_field=mapping(self.magnetic_field),
            line_width1=mapping(self.line_width1),
            line_width2=mapping(self.line_width2),
            line_width3=mapping(self.line_width3),
            line_width4=mapping(self.line_width4),
            line_width5=mapping(self.line_width5),
            line_width6=mapping(self.line_width6),
            line_width_coupling=cls_kwargs.get("line_width_coupling", self.line_width_coupling),
        )

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "SextetGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SextetGeneric`
            Sextet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.channel_span_to_velocity_span(scope.scale),
            quadrupole_split=self.quadrupole_split.channel_span_to_velocity_span(scope.scale),
            ratio13=self.ratio13,
            ratio23=self.ratio23,
            magnetic_field=self.magnetic_field,
            line_width1=self.line_width1.channel_span_to_velocity_span(scope.scale),
            line_width2=self.line_width2.channel_span_to_velocity_span(scope.scale),
            line_width3=self.line_width3.channel_span_to_velocity_span(scope.scale),
            line_width4=self.line_width4.channel_span_to_velocity_span(scope.scale),
            line_width5=self.line_width5.channel_span_to_velocity_span(scope.scale),
            line_width6=self.line_width6.channel_span_to_velocity_span(scope.scale),
            line_width_coupling=self.line_width_coupling,
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "SextetGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SextetGeneric`
            Sextet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=self.isomer_shift.velocity_span_to_channel_span(scope.scale),
            quadrupole_split=self.quadrupole_split.velocity_span_to_channel_span(scope.scale),
            ratio13=self.ratio13,
            ratio23=self.ratio23,
            magnetic_field=self.magnetic_field,
            line_width1=self.line_width1.velocity_span_to_channel_span(scope.scale),
            line_width2=self.line_width2.velocity_span_to_channel_span(scope.scale),
            line_width3=self.line_width3.velocity_span_to_channel_span(scope.scale),
            line_width4=self.line_width4.velocity_span_to_channel_span(scope.scale),
            line_width5=self.line_width5.velocity_span_to_channel_span(scope.scale),
            line_width6=self.line_width6.velocity_span_to_channel_span(scope.scale),
            line_width_coupling=self.line_width_coupling,
        )


class SpectrumGeneric(BaseMossbauerGeneric, Generic[T]):
    """
    Generic class for Mossbauer spectrum.

    Attributes
    ----------
    background : ``T``
        Background of the spectrum
    singlets : ``list`` [:any:`SingletGeneric` [``T`` ]]
        List of singlets
    doublets : ``list`` [:any:`SingletGeneric` [``T`` ]]
        List of doublets
    sextets : ``list`` [:any:`SingletGeneric` [``T`` ]]
        List of sextets

    """

    background: T
    singlets: list[SingletGeneric[T]] = field(default_factory=list)
    doublets: list[DoubletGeneric[T]] = field(default_factory=list)
    sextets: list[SextetGeneric[T]] = field(default_factory=list)

    def map(self, mapping: Callable[[T], T2]) -> "SpectrumGeneric[T2]":
        """
        Map the spectrum using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function

        Returns
        -------
        ``SpectrumGeneric[T2]``
            Mapped spectrum - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.background, SpectrumGeneric)
        return cls(
            background=mapping(self.background),
            singlets=[singlet.map(mapping) for singlet in self.singlets] if self.singlets else [],
            doublets=[doublet.map(mapping) for doublet in self.doublets] if self.doublets else [],
            sextets=[sextet.map(mapping) for sextet in self.sextets] if self.sextets else [],
        )

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "SpectrumGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SpectrumGeneric`
            Spectrum with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            background=self.background,
            singlets=[singlet.to_velocities(scope) for singlet in self.singlets] if self.singlets else [],
            doublets=[doublet.to_velocities(scope) for doublet in self.doublets] if self.doublets else [],
            sextets=[sextet.to_velocities(scope) for sextet in self.sextets] if self.sextets else [],
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "SpectrumGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SpectrumGeneric`
            Spectrum with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            background=self.background,
            singlets=[singlet.to_channels(scope) for singlet in self.singlets] if self.singlets else [],
            doublets=[doublet.to_channels(scope) for doublet in self.doublets] if self.doublets else [],
            sextets=[sextet.to_channels(scope) for sextet in self.sextets] if self.sextets else [],
        )


class SpectroscopeGeneric(BaseMossbauerGeneric, Generic[T]):
    """
    Generic class for Mossbauer spectroscope.

    Attributes
    ----------
    scale : ``T``
        Scale of the spectroscope (channels per mm/s)
    isomer_shift_ref : ``T``
        Reference isomer shift of the spectroscope (position of the energetic zero in channels)

    """

    scale: T
    isomer_shift_ref: T = None

    @property
    def resolution(self):
        """Resolution of the spectroscope (mm/s per channel)."""
        return 1 / self.scale

    def map(self, mapping: Callable[[T], T2]) -> "SpectroscopeGeneric[T2]":
        """
        Map the spectroscope using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function

        Returns
        -------
        ``SpectroscopeGeneric[T2]``
            Mapped spectroscope - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.scale, SpectroscopeGeneric)
        return cls(
            scale=mapping(self.scale),
            isomer_shift_ref=mapping(self.isomer_shift_ref),
        )

    def channel_to_velocity(self, channels: np.ndarray[float] | float) -> np.ndarray[float] | float:
        """Convert channels to velocities using the spectroscope."""
        return channel_to_velocity(channels, self.scale.value, self.isomer_shift_ref.value)

    def velocity_to_channel(self, velocities: np.ndarray[float] | float) -> np.ndarray[float] | float:
        """Convert velocities to channels using the spectroscope."""
        return velocity_to_channel(velocities, self.scale.value, self.isomer_shift_ref.value)


class CalibrationGeneric(BaseMossbauerGeneric, Generic[T]):
    """
    Generic class for Mossbauer calibration.

    Attributes
    ----------
    spectrum : ``SpectrumGeneric[T]``
        Mossbauer spectrum
    scope : ``SpectroscopeGeneric[T]``
        Mossbauer spectroscope

    """

    spectrum: SpectrumGeneric[T]
    scope: SpectroscopeGeneric[T]

    def map(self, mapping: Callable[[T], T2]) -> "CalibrationGeneric[T2]":
        """
        Map the calibration using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function

        Returns
        -------
        ``CalibrationGeneric[T2]``
            Mapped calibration - the exact type depends on the mapping function

        """
        cls = get_typed_cls_from_mapping(mapping, self.spectrum.background, CalibrationGeneric)
        return cls(
            scope=self.scope.map(mapping),
            spectrum=self.spectrum.map(mapping),
        )

    def to_velocities(self) -> "CalibrationGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Returns
        -------
        :any:`CalibrationGeneric`
            Calibration with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_velocities(self.scope.map(lambda x: x.value)),
        )

    def to_channels(self) -> "CalibrationGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Returns
        -------
        :any:`CalibrationGeneric`
            Calibration with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_channels(self.scope.map(lambda x: x.value)),
        )


class AnalysisGeneric(BaseMossbauerGeneric, Generic[T]):
    """
    Generic class for Mossbauer analysis.

    Attributes
    ----------
    spectrum : ``SpectrumGeneric[T]``
        Mossbauer spectrum
    scope : :any:`SpectroscopeComputable`
        Mossbauer spectroscope

    .. note:

        Spectrum type matches the class name logic (e.g. `SpectrumPoint` -> `PointVar`).\
        Regardless of the spectrum type, the scope is always :any:`SpectroscopeComputable`\
        (Analysis does not affect scope).

    """

    spectrum: SpectrumGeneric[T]
    scope: "SpectroscopeComputable"

    def map(self, mapping: Callable[[T], T2]) -> "AnalysisGeneric[T2]":
        """
        Map the analysis using a mapping function.

        Parameters
        ----------
        mapping : ``Callable[[T], T2]``
            Mapping function

        Returns
        -------
        ``AnalysisGeneric[T2]``
            Mapped analysis - the exact type depends on the mapping function

        .. note::

            Mapping is applied to the spectrum only. The scope is always :any:`SpectroscopeComputable`.

        """
        cls = get_typed_cls_from_mapping(mapping, self.spectrum.background, AnalysisGeneric)
        return cls(
            spectrum=self.spectrum.map(mapping),
            scope=self.scope,
        )

    def to_velocities(self) -> "AnalysisGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Returns
        -------
        :any:`AnalysisGeneric`
            Analysis with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_velocities(self.scope),
        )

    def to_channels(self) -> "AnalysisGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Returns
        -------
        :any:`AnalysisGeneric`
            Analysis with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_channels(self.scope),
        )


# ----------------------------------------------------------------------------------------
# Computable classes - has to be included in core due to mapping (otherwise circular import)
# ----------------------------------------------------------------------------------------


# Simple computable models - attributes are directly int, float, or Variable

ComputableVar = Union[int, float, Variable, np.ndarray]


class SingletComputable(SingletGeneric[ComputableVar]):
    """
    Singlet subspectrum with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    amplitude : :any:`ComputableVar`
        Amplitude of the singlet
    isomer_shift : :any:`ComputableVar`
        Isomer shift of the singlet
    line_width1 : :any:`ComputableVar`
        Line width of the singlet

    """

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "SingletGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SingletGeneric`
            Singlet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=channel_span_to_velocity_span(self.isomer_shift, scope.scale),
            line_width1=channel_span_to_velocity_span(self.line_width1, scope.scale),
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "SingletGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`pyMSB.SingletGeneric`
            Singlet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=velocity_span_to_channel_span(self.isomer_shift, scope.scale),
            line_width1=velocity_span_to_channel_span(self.line_width1, scope.scale),
        )


class DoubletComputable(DoubletGeneric[ComputableVar]):
    """
    Doublet subspectrum with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    amplitude : :any:`ComputableVar`
        Amplitude of the doublet
    isomer_shift : :any:`ComputableVar`
        Isomer shift of the doublet
    quadrupole_split : :any:`ComputableVar`
        Quadrupole split of the doublet
    line_width1 : :any:`ComputableVar`
        Line width of the first line
    line_width2 : :any:`ComputableVar`
        Line width of the second line

    """

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "DoubletGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`DoubletGeneric`
            Doublet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=channel_span_to_velocity_span(self.isomer_shift, scope.scale),
            quadrupole_split=channel_span_to_velocity_span(self.quadrupole_split, scope.scale),
            line_width1=channel_span_to_velocity_span(self.line_width1, scope.scale),
            line_width2=channel_span_to_velocity_span(self.line_width2, scope.scale),
            line_width_coupling=self.line_width_coupling,
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "DoubletGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`DoubletGeneric`
            Doublet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=velocity_span_to_channel_span(self.isomer_shift, scope.scale),
            quadrupole_split=velocity_span_to_channel_span(self.quadrupole_split, scope.scale),
            line_width1=velocity_span_to_channel_span(self.line_width1, scope.scale),
            line_width2=velocity_span_to_channel_span(self.line_width2, scope.scale),
            line_width_coupling=self.line_width_coupling,
        )


class SextetComputable(SextetGeneric[ComputableVar]):
    """
    Sextet subspectrum with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    amplitude : :any:`ComputableVar`
        Amplitude of the sextet
    isomer_shift : :any:`ComputableVar`
        Isomer shift of the sextet
    quadrupole_split : :any:`ComputableVar`
        Quadrupole split of the sextet
    ratio13 : :any:`ComputableVar`
        Ratio of the amplitudes of the first and third lines
    ratio23 : :any:`ComputableVar`
        Ratio of the amplitudes of the second and third lines
    magnetic_field : :any:`ComputableVar`
        Magnetic field of the sextet
    line_width1 : :any:`ComputableVar`
        Line width of the first line
    line_width2 : :any:`ComputableVar`
        Line width of the second line
    line_width3 : :any:`ComputableVar`
        Line width of the third line
    line_width4 : :any:`ComputableVar`
        Line width of the fourth line
    line_width5 : :any:`ComputableVar`
        Line width of the fifth line
    line_width6 : :any:`ComputableVar`
        Line width of the sixth line

    """

    def to_velocities(self, scope: "SpectroscopeGeneric[T]") -> "SextetGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SextetGeneric`
            Sextet with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=channel_span_to_velocity_span(self.isomer_shift, scope.scale),
            quadrupole_split=channel_span_to_velocity_span(self.quadrupole_split, scope.scale),
            ratio13=self.ratio13,
            ratio23=self.ratio23,
            magnetic_field=self.magnetic_field,
            line_width1=channel_span_to_velocity_span(self.line_width1, scope.scale),
            line_width2=channel_span_to_velocity_span(self.line_width2, scope.scale),
            line_width3=channel_span_to_velocity_span(self.line_width3, scope.scale),
            line_width4=channel_span_to_velocity_span(self.line_width4, scope.scale),
            line_width5=channel_span_to_velocity_span(self.line_width5, scope.scale),
            line_width6=channel_span_to_velocity_span(self.line_width6, scope.scale),
            line_width_coupling=self.line_width_coupling,
        )

    def to_channels(self, scope: "SpectroscopeGeneric[T]") -> "SextetGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Parameters
        ----------
        scope : :any:`SpectroscopeGeneric`
            Spectroscope

        Returns
        -------
        :any:`SextetGeneric`
            Sextet with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            name=self.name,
            amplitude=self.amplitude,
            isomer_shift=velocity_span_to_channel_span(self.isomer_shift, scope.scale),
            quadrupole_split=velocity_span_to_channel_span(self.quadrupole_split, scope.scale),
            ratio13=self.ratio13,
            ratio23=self.ratio23,
            magnetic_field=self.magnetic_field,
            line_width1=velocity_span_to_channel_span(self.line_width1, scope.scale),
            line_width2=velocity_span_to_channel_span(self.line_width2, scope.scale),
            line_width3=velocity_span_to_channel_span(self.line_width3, scope.scale),
            line_width4=velocity_span_to_channel_span(self.line_width4, scope.scale),
            line_width5=velocity_span_to_channel_span(self.line_width5, scope.scale),
            line_width6=velocity_span_to_channel_span(self.line_width6, scope.scale),
            line_width_coupling=self.line_width_coupling,
        )


class SpectrumComputable(SpectrumGeneric[ComputableVar]):
    """
    Mossbauer spectrum with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    background : :any:`ComputableVar`
        Background of the spectrum
    singlets : ``list`` [:any:`SingletComputable`]
        List of singlets
    doublets : ``list`` [:any:`SingletComputable`]
        List of doublets
    sextets : ``list`` [:any:`SingletComputable`]
        List of sextets

    """

    pass


class SpectroscopeComputable(SpectroscopeGeneric[ComputableVar]):
    """
    Mossbauer spectroscope with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    scale : :any:`ComputableVar`
        Scale of the spectroscope (channels per mm/s)
    isomer_shift_ref : :any:`ComputableVar`
        Reference isomer shift of the spectroscope (position of the energetic zero in channels)

    """

    def channel_to_velocity(self, channels: np.ndarray[float] | float) -> np.ndarray[float] | float:
        """Convert channels to velocities using the spectroscope."""
        return channel_to_velocity(channels, self.scale, self.isomer_shift_ref)

    def velocity_to_channel(self, velocities: np.ndarray[float] | float) -> np.ndarray[float] | float:
        """Convert velocities to channels using the spectroscope."""
        return velocity_to_channel(velocities, self.scale, self.isomer_shift_ref)


class CalibrationComputable(CalibrationGeneric[ComputableVar]):
    """
    Mossbauer calibration with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    spectrum : :any:`SpectrumComputable`
        Mossbauer spectrum
    scope : :any:`SpectroscopeComputable`
        Mossbauer spectroscope

    """

    def to_velocities(self) -> "CalibrationGeneric[T]":
        """
        Convert parameters from channels to velocities using the spectroscope.

        Returns
        -------
        :any:`CalibrationGeneric`
            Calibration with parameters converted to velocities

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_velocities(self.scope),
        )

    def to_channels(self) -> "CalibrationGeneric[T]":
        """
        Convert parameters from velocities to channels using the spectroscope.

        Returns
        -------
        :any:`CalibrationGeneric`
            Calibration with parameters converted to channels

        """
        cls = self.__class__
        return cls(
            scope=self.scope,
            spectrum=self.spectrum.to_channels(self.scope),
        )


class AnalysisComputable(AnalysisGeneric[ComputableVar]):
    """
    Mossbauer analysis with attributes as computable variables.

    Primary use is as input to :any:`pyMSB.common` functions.

    Attributes
    ----------
    spectrum : :any:`SpectrumComputable`
        Mossbauer spectrum
    scope : :any:`SpectroscopeComputable`
        Mossbauer spectroscope

    """

    pass


SpectrumT = TypeVar("SpectrumT", bound=SpectrumGeneric)
SpectroscopeT = TypeVar("SpectroscopeT", bound=SpectroscopeGeneric)
CalibrationT = TypeVar("CalibrationT", bound=CalibrationGeneric)
AnalysisT = TypeVar("AnalysisT", bound=AnalysisGeneric)


COMPUTABLE_MAPPING = {
    SingletGeneric: SingletComputable,
    DoubletGeneric: DoubletComputable,
    SextetGeneric: SextetComputable,
    SpectrumGeneric: SpectrumComputable,
    SpectroscopeGeneric: SpectroscopeComputable,
    CalibrationGeneric: CalibrationComputable,
    AnalysisGeneric: AnalysisComputable,
}
