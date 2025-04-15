import copy

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from pyMSB.common import spectrum_func
from pyMSB.computation_models.base import AAnalysis, ACalibration
from pyMSB.models import (
    AnalysisPoint,
    CalibrationPoint,
    DoubletSpecs,
    LineWidthCoupling,
    PointVar,
    SextetSpecs,
    SingletSpecs,
    SpecsVar,
    SpectroscopeComputable,
    SpectroscopeGeometry,
    SpectroscopeSpecs,
    SpectrumSpecs,
)


class LSQModelMixin:
    """Mixin class for Least Squares fitting models."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)

        self._fit_func = None
        self._p0 = None

    def _initialize(self):
        """Initialize the model."""
        self._fit_func = self._create_fit_func(self._specs.spectrum, self._specs.scope, self._geometry)
        self._p0 = self._flatten_fit_params(self._specs.spectrum) + self._flatten_fit_params(self._specs.scope)

    def _fit_(self) -> None:
        """Fit the model to the data."""
        return

    def _flatten_fit_params(self, obj: SpectrumSpecs | SpectroscopeSpecs | SingletSpecs | DoubletSpecs | SextetSpecs):
        match obj:
            case DoubletSpecs():
                return self._flatten_doublet_fit_params(obj)
            case SextetSpecs():
                return self._flatten_sextet_fit_params(obj)
        params = []
        for _attr, item in obj.__dict__.items():
            match item:
                case SpecsVar(value=int() | float(), is_fixed=False):
                    params.append(item.value)
                case list():
                    for element in item:
                        params.extend(self._flatten_fit_params(element))
        return params

    @staticmethod
    def _flatten_doublet_fit_params(doublet: DoubletSpecs) -> list:
        params = []
        params += [doublet.amplitude.value] if not doublet.amplitude.is_fixed else []
        params += [doublet.isomer_shift.value] if not doublet.isomer_shift.is_fixed else []
        params += [doublet.quadrupole_split.value] if not doublet.quadrupole_split.is_fixed else []
        params += [doublet.line_width1.value] if not doublet.line_width1.is_fixed else []
        if doublet.line_width_coupling == LineWidthCoupling.UNCOUPLED:
            params += [doublet.line_width2.value] if not doublet.line_width2.is_fixed else []
        return params

    @staticmethod
    def _flatten_sextet_fit_params(sextet: SextetSpecs) -> list:
        params = []
        params += [sextet.amplitude.value] if not sextet.amplitude.is_fixed else []
        params += [sextet.isomer_shift.value] if not sextet.isomer_shift.is_fixed else []
        params += [sextet.quadrupole_split.value] if not sextet.quadrupole_split.is_fixed else []
        params += [sextet.ratio13.value] if not sextet.ratio13.is_fixed else []
        params += [sextet.ratio23.value] if not sextet.ratio23.is_fixed else []
        params += [sextet.magnetic_field.value] if not sextet.magnetic_field.is_fixed else []
        params += [sextet.line_width1.value] if not sextet.line_width1.is_fixed else []
        match sextet.line_width_coupling:
            case LineWidthCoupling.UNCOUPLED:
                params += [sextet.line_width2.value] if not sextet.line_width2.is_fixed else []
                params += [sextet.line_width3.value] if not sextet.line_width3.is_fixed else []
                params += [sextet.line_width4.value] if not sextet.line_width4.is_fixed else []
                params += [sextet.line_width5.value] if not sextet.line_width5.is_fixed else []
                params += [sextet.line_width6.value] if not sextet.line_width6.is_fixed else []
            case LineWidthCoupling.COUPLED_PAIRS:
                params += [sextet.line_width2.value] if not sextet.line_width2.is_fixed else []
                params += [sextet.line_width3.value] if not sextet.line_width3.is_fixed else []
        return params

    def _rebuild_from_fit_params(
        self,
        spectrum: SpectrumSpecs,
        spectroscope: SpectroscopeComputable,
        params: tuple | list,
        sigmas: tuple | list = None,
    ):
        spectrum_copy = copy.deepcopy(spectrum)
        if sigmas is not None:
            var_list = list(map(lambda param, sigma: PointVar(value=param, sigma=sigma), params, sigmas))
            var_iterator = iter(var_list)
            self._rebuild_obj_from_fit_params(spectrum_copy, var_iterator)
            spectrum_copy = spectrum_copy.map(lambda x: PointVar(x.value) if isinstance(x, SpecsVar) else x)
        else:
            var_list = list(map(lambda param: SpecsVar(param), params))
            var_iterator = iter(var_list)
            self._rebuild_obj_from_fit_params(spectrum_copy, var_iterator)
            spectrum_copy = spectrum_copy.map(lambda x: x.value)

        return spectrum_copy, spectroscope

    def _rebuild_obj_from_fit_params(
        self, obj: SpectrumSpecs | SpectroscopeSpecs | SingletSpecs | DoubletSpecs | SextetSpecs, var_iterator: iter
    ):
        match obj:
            case DoubletSpecs():
                self._rebuild_doublet_from_fit_params(obj, var_iterator)
                return
            case SextetSpecs():
                self._rebuild_sextet_from_fit_params(obj, var_iterator)
                return
        for attr, item in obj.__dict__.items():
            match item:
                case SpecsVar(value=int() | float(), is_fixed=False):
                    setattr(obj, attr, next(var_iterator))  # Replace non-fixed parameter value
                case list():
                    for element in item:
                        self._rebuild_obj_from_fit_params(element, var_iterator)

    @staticmethod
    def _rebuild_doublet_from_fit_params(doublet: DoubletSpecs, var_iterator: iter):
        doublet.amplitude = next(var_iterator) if not doublet.amplitude.is_fixed else doublet.amplitude
        doublet.isomer_shift = next(var_iterator) if not doublet.isomer_shift.is_fixed else doublet.isomer_shift
        doublet.quadrupole_split = (
            next(var_iterator) if not doublet.quadrupole_split.is_fixed else doublet.quadrupole_split
        )
        doublet.line_width1 = next(var_iterator) if not doublet.line_width1.is_fixed else doublet.line_width1
        if doublet.line_width_coupling in [LineWidthCoupling.COUPLED_PAIRS, LineWidthCoupling.COUPLED]:
            doublet.line_width2 = doublet.line_width1
        else:
            doublet.line_width2 = next(var_iterator) if not doublet.line_width2.is_fixed else doublet.line_width2

    @staticmethod
    def _rebuild_sextet_from_fit_params(sextet: SextetSpecs, var_iterator: iter):
        sextet.amplitude = next(var_iterator) if not sextet.amplitude.is_fixed else sextet.amplitude
        sextet.isomer_shift = next(var_iterator) if not sextet.isomer_shift.is_fixed else sextet.isomer_shift
        sextet.quadrupole_split = (
            next(var_iterator) if not sextet.quadrupole_split.is_fixed else sextet.quadrupole_split
        )
        sextet.ratio13 = next(var_iterator) if not sextet.ratio13.is_fixed else sextet.ratio13
        sextet.ratio23 = next(var_iterator) if not sextet.ratio23.is_fixed else sextet.ratio23
        sextet.magnetic_field = next(var_iterator) if not sextet.magnetic_field.is_fixed else sextet.magnetic_field
        sextet.line_width1 = next(var_iterator) if not sextet.line_width1.is_fixed else sextet.line_width1
        match sextet.line_width_coupling:
            case LineWidthCoupling.UNCOUPLED:
                sextet.line_width2 = next(var_iterator) if not sextet.line_width2.is_fixed else sextet.line_width2
                sextet.line_width3 = next(var_iterator) if not sextet.line_width3.is_fixed else sextet.line_width3
                sextet.line_width4 = next(var_iterator) if not sextet.line_width4.is_fixed else sextet.line_width4
                sextet.line_width5 = next(var_iterator) if not sextet.line_width5.is_fixed else sextet.line_width5
                sextet.line_width6 = next(var_iterator) if not sextet.line_width6.is_fixed else sextet.line_width6
            case LineWidthCoupling.COUPLED_PAIRS:
                sextet.line_width2 = next(var_iterator) if not sextet.line_width2.is_fixed else sextet.line_width2
                sextet.line_width3 = next(var_iterator) if not sextet.line_width3.is_fixed else sextet.line_width3
                sextet.line_width4 = sextet.line_width3
                sextet.line_width5 = sextet.line_width2
                sextet.line_width6 = sextet.line_width1
            case LineWidthCoupling.COUPLED:
                sextet.line_width2 = sextet.line_width1
                sextet.line_width3 = sextet.line_width1
                sextet.line_width4 = sextet.line_width1
                sextet.line_width5 = sextet.line_width1
                sextet.line_width6 = sextet.line_width1

    def _create_fit_func(
        self,
        spectrum_estimate: SpectrumSpecs,
        spectroscope_estimate: SpectroscopeComputable | SpectroscopeSpecs,
        geometry: SpectroscopeGeometry,
    ):
        def func(x, *args):
            spectrum, spectroscope = self._rebuild_from_fit_params(spectrum_estimate, spectroscope_estimate, args)
            return spectrum_func(channels=x, spectrum=spectrum, spectroscope=spectroscope, geometry=geometry)

        return func


class LSQModel(LSQModelMixin, AAnalysis):
    """
    Least Squares fitting model for Analysis.

    Attributes
    ----------
    specs : :any:`AnalysisSpecs`
        The specifications for the spectrum (user estimate).
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "LSQ"
    results_type = AnalysisPoint

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)  # a single call is enough to invoke all parent constructors

        self._fitted_params = None
        self._covariance_matrix = None

    def _validate_specifications(self):
        super()._validate_specifications()
        self._analyze_line_width_physics(self._specs)

    def _clear_attributes(self):
        super()._clear_attributes()
        self._fit_func = None
        self._p0 = None
        self._fitted_params = None
        self._covariance_matrix = None

    def _fit(self) -> None:
        self._fitted_params, self._covariance_matrix, *_ = curve_fit(
            self._fit_func, self._channels, self._counts, p0=self._p0
        )

    def _create_results(self) -> AnalysisPoint:
        sigmas = np.sqrt(np.diag(self._covariance_matrix))
        spectrum_res, scope_res = self._rebuild_from_fit_params(
            self._specs.spectrum, self._specs.scope, self._fitted_params, sigmas
        )
        return AnalysisPoint(spectrum=spectrum_res, scope=scope_res)

    def predict(self, channels: np.ndarray | list[int]) -> np.ndarray[float]:
        """
        Return the predicted spectrum evaluated at the given channels.

        Attributes
        ----------
        channels : np.ndarray|list[int]
            The channels at which to evaluate the spectrum.

        Returns
        -------
        np.ndarray[float]
            The predicted spectrum evaluated at the given channels.

        """
        return spectrum_func(
            channels,
            spectrum=self._results.spectrum.map(lambda x: x.value),
            spectroscope=self._specs.scope,
            geometry=self._geometry,
        )


class LSQCalibrationModel(LSQModelMixin, ACalibration):
    """
    Least Squares fitting model for Calibration.

    Attributes
    ----------
    specs : :any:`CalibrationSpecs`
        The specifications for the calibration (user estimate).
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "LSQ"
    results_type = CalibrationPoint

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        super().__init__(*args, **kwargs)  # a single call is enough to invoke all parent constructors

        self._fitted_params = None
        self._covariance_matrix = None

    def _validate_specifications(self):
        super()._validate_specifications()
        self._analyze_line_width_physics(self._specs)

    def _clear_attributes(self):
        super()._clear_attributes()
        self._fit_func = None
        self._p0 = None
        self._fitted_params = None
        self._covariance_matrix = None

    def _fit(self) -> None:
        self._fitted_params, self._covariance_matrix, *_ = curve_fit(
            self._fit_func, self._channels, self._counts, p0=self._p0
        )

    def _create_results(self) -> CalibrationPoint:
        sigmas = np.sqrt(np.diag(self._covariance_matrix))
        spectrum_res, scope_res = self._rebuild_from_fit_params(
            self._specs.spectrum, self._specs.scope, self._fitted_params, sigmas
        )
        return CalibrationPoint(spectrum=spectrum_res, scope=scope_res)

    def predict(self, channels: ArrayLike) -> np.ndarray[float]:
        """
        Return the predicted spectrum evaluated at the given channels.

        Attributes
        ----------
        channels : ArrayLike
            The channels at which to evaluate the spectrum.

        Returns
        -------
        np.ndarray[float]
            The predicted spectrum evaluated at the given channels.

        """
        return spectrum_func(
            channels=channels,
            spectrum=self._results.spectrum.map(lambda x: x.value),
            spectroscope=self._specs.scope.map(lambda x: x.value),
            geometry=self._geometry,
        )

    def _rebuild_from_fit_params(
        self,
        spectrum: SpectrumSpecs,
        spectroscope: SpectroscopeSpecs,
        params: tuple | list,
        sigmas: tuple | list = None,
    ):
        spectrum_copy = copy.deepcopy(spectrum)
        spectroscope_copy = copy.deepcopy(spectroscope)
        if sigmas is not None:
            var_list = list(map(lambda param, sigma: PointVar(value=param, sigma=sigma), params, sigmas))
            var_iterator = iter(var_list)
            self._rebuild_obj_from_fit_params(spectrum_copy, var_iterator)
            self._rebuild_obj_from_fit_params(spectroscope_copy, var_iterator)
            spectrum_copy = spectrum_copy.map(lambda x: PointVar(x.value) if isinstance(x, SpecsVar) else x)
            spectroscope_copy = spectroscope_copy.map(lambda x: PointVar(x.value) if isinstance(x, SpecsVar) else x)
        else:
            var_list = list(map(lambda param: SpecsVar(param), params))
            var_iterator = iter(var_list)
            self._rebuild_obj_from_fit_params(spectrum_copy, var_iterator)
            self._rebuild_obj_from_fit_params(spectroscope_copy, var_iterator)
            spectrum_copy = spectrum_copy.map(lambda x: x.value)
            spectroscope_copy = spectroscope_copy.map(lambda x: x.value)

        return spectrum_copy, spectroscope_copy
