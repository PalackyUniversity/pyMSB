import abc
from typing import Self

import numpy as np

from pyMSB.exceptions import MossbauerExc, PhysicsValidationExc, ValidationExc
from pyMSB.models import (
    AnalysisSpecs,
    AnalysisT,
    CalibrationSpecs,
    CalibrationT,
    LineWidthCoupling,
    SpectroscopeGeometry,
)
from pyMSB.statistics import is_transmission_spectrum

# specs - specifications defining the structure of the model (subspectra) and used as initial values or priors.
# config - configuration defining the computation process (e.g. number of iterations, num of samples, ...).


class AModel(abc.ABC):
    name: str
    results_type: type[CalibrationT | AnalysisT]

    def __init__(self, specs, geometry, verbose: bool = False, strict: bool = True):
        """Initialize the model."""
        self.__verbose = verbose
        self.__strict = strict

        # params
        self._specs = specs
        self._geometry = geometry

        # attributes
        self._channels: np.ndarray | None = None
        self._counts: np.ndarray | None = None
        self._results = None

        self._validate_specifications()

        if self.__strict:
            self._validate_physics()

    def _initialize(self):
        """
        Initialize the model.

        For example, build the bayesian model, set the initial parameters, etc.
        This method is not an abstract method because it is not required for all models.
        """
        return

    def _clear_attributes(self):
        self._channels = None
        self._counts = None
        self._results: CalibrationT | AnalysisT

    @abc.abstractmethod
    def _validate_specifications(self):
        """Validate specifications and raise an error if invalid."""
        pass

    def _validate_data(self, channels, counts):
        """Validate the input data and raise an error if invalid."""
        if len(channels) != len(counts):
            raise ValidationExc("Channels and counts must have the same length.")

        if (is_transmission_spectrum(counts) and (self._geometry != SpectroscopeGeometry.TRANSMISSION)) or (
            not is_transmission_spectrum(counts) and (self._geometry == SpectroscopeGeometry.TRANSMISSION)
        ):
            raise ValidationExc(f"Invalid geometry – spectrum seems to be a transmission, but {self._geometry} is set.")

    def _validate_physics(self):
        """Validate if the parameters are physically meaningful. Raise an error with first problem if invalid."""
        for problem in self.analyze_physics(self._specs):
            raise PhysicsValidationExc(problem)

    @classmethod
    def analyze_physics(cls, specs) -> list[str]:
        """Analyze if the parameters are physically meaningful. Return a list of problems if any."""
        problems = []
        problems.extend(cls._analyze_line_width_physics(specs))
        return problems

    @staticmethod
    def _analyze_line_width_physics(specs) -> list[str]:
        problems = []
        spectrum_specs = specs.spectrum
        for doublet in spectrum_specs.doublets:
            if doublet.line_width_coupling == LineWidthCoupling.UNCOUPLED:
                continue
            if doublet.line_width1.is_fixed != doublet.line_width2.is_fixed:
                problems.append(
                    f"Coupled line widths must share the same fixed status. "
                    f"Criteria not met in doublet '{doublet.name}'"
                )
        for sextet in spectrum_specs.sextets:
            match sextet.line_width_coupling:
                case LineWidthCoupling.UNCOUPLED:
                    continue
                case LineWidthCoupling.COUPLED_PAIRS:
                    if sextet.line_width1.is_fixed != sextet.line_width6.is_fixed:
                        problems.append(
                            f"Coupled line widths must share the same fixed status. "
                            f"Criteria not met in sextet '{sextet.name}' at line 1 and 6."
                        )
                    if sextet.line_width2.is_fixed != sextet.line_width5.is_fixed:
                        problems.append(
                            f"Coupled line widths must share the same fixed status. "
                            f"Criteria not met in sextet '{sextet.name}' at line 2 and 5."
                        )
                    if sextet.line_width3.is_fixed != sextet.line_width4.is_fixed:
                        problems.append(
                            f"Coupled line widths must share the same fixed status. "
                            f"Criteria not met in sextet '{sextet.name}' at line 3 and 4."
                        )
                case LineWidthCoupling.COUPLED:
                    line_widths = [
                        sextet.line_width1,
                        sextet.line_width2,
                        sextet.line_width3,
                        sextet.line_width4,
                        sextet.line_width5,
                        sextet.line_width6,
                    ]
                    if not all(line_widths[0].is_fixed == lw.is_fixed for lw in line_widths):
                        problems.append(
                            f"Coupled line widths must share the same fixed status. "
                            f"Criteria not met in sextet '{sextet.name}'."
                        )
        return problems

    @abc.abstractmethod
    def _fit(self):
        """
        Estimate and store model attributes from the estimated parameters and provided data.

        For example, in a Bayesian model, this method would sample the posterior distribution.
        For converting the model specific results to ._results the _create_results() method is used in .fit() method.
        """
        pass

    def fit(self, channels: np.ndarray[int], counts: np.ndarray[int]) -> Self:
        """
        Fit the model to the provided data.

        Parameters
        ----------
        channels : np.ndarray[int]
            The channels of the spectrum.
        counts : np.ndarray[int]
            The counts of the spectrum.

        Returns
        -------
        Self
            The fitted estimator.

        """
        # 1) clear any prior attributes
        self._clear_attributes()
        # 2) validate the input data;
        self._validate_data(channels, counts)
        # 3) store the input data;
        self._channels = channels
        self._counts = counts
        # 4) Initialize the model
        self._initialize()
        # 5) estimate and store model attributes from the estimated parameters and provided data
        self._fit()
        # 6) Convert the results to a more user-friendly format in self._results
        self._results = self._create_results()
        # 7) return the now fitted estimator to facilitate method chaining.
        return self

    @abc.abstractmethod
    def _create_results(self):
        """Convert the results to a Results format."""
        pass

    @property
    def results(self) -> CalibrationT | AnalysisT:
        if self._results is None:
            raise MossbauerExc("Results are not available. Run `fit()` first.")
        return self._results

    @abc.abstractmethod
    def predict(self, channels: np.ndarray | list[int]) -> np.ndarray[float]:
        pass


class AAnalysis(AModel, abc.ABC):
    results_type: type[AnalysisT]

    def _validate_specifications(self):
        """Validate specifications and raise an error if invalid."""
        if not isinstance(self._specs, AnalysisSpecs):
            raise ValidationExc(f"specs must be an instance of AnalysisSpecs, instead is {type(self._specs)}.")

    @abc.abstractmethod
    def _create_results(self) -> AnalysisT:
        pass


class ACalibration(AModel, abc.ABC):
    results_type: type[CalibrationT]

    def _validate_specifications(self):
        """Validate specifications and raise an error if invalid."""
        if not isinstance(self._specs, CalibrationSpecs):
            raise ValidationExc(f"specs must be an instance of CalibrationSpecs, instead is {type(self._specs)}.")

    @classmethod
    def analyze_physics(cls, specs: CalibrationSpecs) -> list[str]:
        """Analyze if the parameters are physically meaningful. Return a list of problems if any."""
        problems = super().analyze_physics(specs)
        if len(specs.spectrum.sextets) <= 0:
            problems.append("Calibration procedure requires at least one sextet to be present in the spectrum.")
        if not any(sextet.magnetic_field.is_fixed for sextet in specs.spectrum.sextets):
            problems.append("At least one sextet in the spectrum must have a fixed magnetic field.")
        subspectra = specs.spectrum.singlets + specs.spectrum.doublets + specs.spectrum.sextets
        if not any(subspectrum.isomer_shift.is_fixed for subspectrum in subspectra):
            problems.append("At least one subspectrum must have a fixed isomer shift.")
        if specs.scope.scale.is_fixed:
            problems.append("Fixing scale is not recommended for calibration.")
        if specs.scope.isomer_shift_ref.is_fixed:
            problems.append("Fixing isomer shift reference is not recommended for calibration.")
        return problems

    @abc.abstractmethod
    def _create_results(self) -> CalibrationT:
        pass
