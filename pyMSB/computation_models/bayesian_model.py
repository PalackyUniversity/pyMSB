"""
Two Bayesian models enable fitting Mössbauer spectra for both Calibration and Analysis respectively.

Two methods of estimation are available:
    - Maximum a posteriori - yielding point estimate without uncertainty
    - Full Bayesian Inference - yielding posterior distributions of all fit parameters



    .....TODO.......
"""

import abc

import arviz as az
import numpy as np
import pymc as pm
from pytensor import Variable

from pyMSB.common import spectrum_func
from pyMSB.computation_models.base import AAnalysis, ACalibration, AModel
from pyMSB.exceptions import ValidationExc
from pyMSB.models import (
    AnalysisBayes,
    AnalysisPoint,
    BayesVar,
    CalibrationBayes,
    CalibrationPoint,
    DoubletRVs,
    DoubletSpecs,
    LineWidthCoupling,
    PointVar,
    SextetRVs,
    SextetSpecs,
    SingletRVs,
    SingletSpecs,
    SpecsVar,
    SpectroscopeRVs,
    SpectroscopeSpecs,
    SpectrumRVs,
    SpectrumSpecs,
)


class BayesianModelMixin(AModel, abc.ABC):
    """Mixin class for Bayesian models."""

    def __init__(self, *args, **kwargs):
        """Initialize the Bayesian model mixin."""
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.model: pm.Model = pm.Model()

    def _find_map(self, *args, **kwargs) -> dict[str, int | float]:
        """Find Maximum a Posteriori estimate of the model parameters."""
        with self.model:
            map_estimate = pm.find_MAP(*args, **kwargs)
        return map_estimate

    def _sample(self, *args, **kwargs) -> az.InferenceData:
        """Sample from the posterior distribution of the model parameters."""
        with self.model:
            idata = pm.sample(*args, **kwargs)
        return idata

    def _build_spectrum_rvs(self, spectrum_specs: SpectrumSpecs):
        spectrum_rvs: SpectrumRVs = SpectrumRVs(background=None, singlets=[], doublets=[], sextets=[])
        spectrum_rvs.background = self._build_background_rvs(spectrum_specs.background)
        for k, singlet in enumerate(spectrum_specs.singlets):
            spectrum_rvs.singlets.append(self._build_singlet_rvs(singlet, k))
        for k, doublet in enumerate(spectrum_specs.doublets):
            spectrum_rvs.doublets.append(self._build_doublet_rvs(doublet, k))
        for k, sextet in enumerate(spectrum_specs.sextets):
            spectrum_rvs.sextets.append(self._build_sextet_rvs(sextet, k))
        return spectrum_rvs

    def _build_background_rvs(self, bckg: SpecsVar) -> Variable:
        with self.model:
            if bckg.is_fixed:
                r = pm.Deterministic("bckg", pm.math.constant(bckg.value))
            else:
                bckg_coeff = pm.Normal("bckg_coeff", mu=100, sigma=1)
                r = pm.Deterministic("bckg", bckg.value * 0.01 * bckg_coeff)
        return r

    def _build_amplitude_rvs(self, rv_name: str, amplitude: SpecsVar):
        with self.model:
            if amplitude.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(amplitude.value))
            else:
                # sigma is 20 or at least 3*std of noise
                # sigma=max(20, 100 * 3 * Stats.get_noise_std(self.counts) / singlet.estimates['amp'])
                amplitude_coeff = pm.Normal(f"{rv_name}_coeff", mu=100, sigma=20)
                return pm.Deterministic(rv_name, amplitude.value * 0.01 * amplitude_coeff)

    def _build_isomer_shift_rvs(self, rv_name: str, isomer_shift: SpecsVar):
        with self.model:
            if isomer_shift.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(isomer_shift.value))
            else:
                return pm.Normal(rv_name, mu=isomer_shift.value, sigma=len(self._channels) / 100)

    def _build_line_width_rvs(self, rv_name: str, line_width: SpecsVar):
        with self.model:
            if line_width.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(line_width.value))
            else:
                return pm.LogNormal(rv_name, mu=np.log(line_width.value), sigma=np.log(line_width.value * 0.2))

    def _build_coupled_line_widths_rvs(self, rv_common_name: str, rv_names: list[str], line_widths: list[SpecsVar]):
        if len(set(lw.is_fixed for lw in line_widths)) > 1:
            raise ValidationExc("All line widths must be fixed or all must be variable.")

        with self.model:
            if line_widths[0].is_fixed:
                line_width = pm.Deterministic(rv_common_name, pm.math.constant(line_widths[0].value))
            else:
                line_width = pm.LogNormal(
                    rv_common_name, mu=np.log(line_widths[0].value), sigma=np.log(line_widths[0].value * 0.2)
                )
            return tuple(pm.Deterministic(rv_name, line_width) for rv_name in rv_names)

    def _build_quadrupole_split_rvs(self, rv_name: str, quadrupole_split: SpecsVar):
        with self.model:
            if quadrupole_split.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(quadrupole_split.value))
            else:
                return pm.Normal(rv_name, mu=quadrupole_split.value, sigma=len(self._channels) / 100)

    def _build_ratio_rvs(self, rv_name: str, ratio: SpecsVar):
        with self.model:
            if ratio.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(ratio.value))
            else:
                return pm.Normal(rv_name, mu=ratio.value, sigma=ratio.value * 0.2)

    def _build_magnetic_field_rvs(self, rv_name: str, magnetic_field: SpecsVar):
        with self.model:
            if magnetic_field.is_fixed:
                return pm.Deterministic(rv_name, pm.math.constant(magnetic_field.value))
            else:
                return pm.Normal(rv_name, mu=magnetic_field.value, sigma=1)

    def _build_singlet_rvs(self, singlet: SingletSpecs, k: int):
        singlet_rvs = SingletRVs(name=singlet.name, amplitude=None, isomer_shift=None, line_width1=None)
        singlet_rvs.amplitude = self._build_amplitude_rvs(f"sg{k}_amplitude", singlet.amplitude)
        singlet_rvs.isomer_shift = self._build_isomer_shift_rvs(f"sg{k}_isomer_shift", singlet.isomer_shift)
        singlet_rvs.line_width1 = self._build_line_width_rvs(f"sg{k}_line_width", singlet.line_width1)
        return singlet_rvs

    def _build_doublet_rvs(self, doublet: DoubletSpecs, k: int) -> DoubletRVs:
        doublet_rvs = DoubletRVs(
            name=doublet.name,
            amplitude=None,
            isomer_shift=None,
            quadrupole_split=None,
            line_width1=None,
            line_width2=None,
        )

        doublet_rvs.amplitude = self._build_amplitude_rvs(f"db{k}_amplitude", doublet.amplitude)
        doublet_rvs.isomer_shift = self._build_isomer_shift_rvs(f"db{k}_isomer_shift", doublet.isomer_shift)
        doublet_rvs.quadrupole_split = self._build_quadrupole_split_rvs(
            f"db{k}_quadrupole_split", doublet.quadrupole_split
        )
        match doublet.line_width_coupling:
            case LineWidthCoupling.UNCOUPLED:
                doublet_rvs.line_width1 = self._build_line_width_rvs(f"db{k}_line_width1", doublet.line_width1)
                doublet_rvs.line_width2 = self._build_line_width_rvs(f"db{k}_line_width2", doublet.line_width2)
            case LineWidthCoupling.COUPLED_PAIRS | LineWidthCoupling.COUPLED:
                doublet_rvs.line_width1, doublet_rvs.line_width2 = self._build_coupled_line_widths_rvs(
                    f"db{k}_line_width12",
                    [f"db{k}_line_width1", f"db{k}_line_width2"],
                    [doublet.line_width1, doublet.line_width2],
                )
        return doublet_rvs

    def _build_sextet_rvs(self, sextet: SextetSpecs, k: int):
        sextet_rvs = SextetRVs(
            name=sextet.name,
            amplitude=None,
            isomer_shift=None,
            quadrupole_split=None,
            ratio13=None,
            ratio23=None,
            magnetic_field=None,
            line_width1=None,
            line_width2=None,
            line_width3=None,
            line_width4=None,
            line_width5=None,
            line_width6=None,
        )

        sextet_rvs.amplitude = self._build_amplitude_rvs(f"sx{k}_amplitude", sextet.amplitude)
        sextet_rvs.isomer_shift = self._build_isomer_shift_rvs(f"sx{k}_isomer_shift", sextet.isomer_shift)
        sextet_rvs.quadrupole_split = self._build_quadrupole_split_rvs(
            f"sx{k}_quadrupole_split", sextet.quadrupole_split
        )
        sextet_rvs.ratio13 = self._build_ratio_rvs(f"sx{k}_ratio13", sextet.ratio13)
        sextet_rvs.ratio23 = self._build_ratio_rvs(f"sx{k}_ratio23", sextet.ratio23)
        sextet_rvs.magnetic_field = self._build_magnetic_field_rvs(f"sx{k}_magnetic_field", sextet.magnetic_field)

        match sextet.line_width_coupling:
            case LineWidthCoupling.UNCOUPLED:
                sextet_rvs.line_width1 = self._build_line_width_rvs(f"sx{k}_line_width1", sextet.line_width1)
                sextet_rvs.line_width2 = self._build_line_width_rvs(f"sx{k}_line_width2", sextet.line_width2)
                sextet_rvs.line_width3 = self._build_line_width_rvs(f"sx{k}_line_width3", sextet.line_width3)
                sextet_rvs.line_width4 = self._build_line_width_rvs(f"sx{k}_line_width4", sextet.line_width4)
                sextet_rvs.line_width5 = self._build_line_width_rvs(f"sx{k}_line_width5", sextet.line_width5)
                sextet_rvs.line_width6 = self._build_line_width_rvs(f"sx{k}_line_width6", sextet.line_width6)
            case LineWidthCoupling.COUPLED_PAIRS:
                sextet_rvs.line_width1, sextet_rvs.line_width6 = self._build_coupled_line_widths_rvs(
                    f"sx{k}_line_width16",
                    [f"sx{k}_line_width1", f"sx{k}_line_width6"],
                    [sextet.line_width1, sextet.line_width6],
                )
                sextet_rvs.line_width2, sextet_rvs.line_width5 = self._build_coupled_line_widths_rvs(
                    f"sx{k}_line_width25",
                    [f"sx{k}_line_width2", f"sx{k}_line_width5"],
                    [sextet.line_width2, sextet.line_width5],
                )
                sextet_rvs.line_width3, sextet_rvs.line_width4 = self._build_coupled_line_widths_rvs(
                    f"sx{k}_line_width34",
                    [f"sx{k}_line_width3", f"sx{k}_line_width4"],
                    [sextet.line_width3, sextet.line_width4],
                )
            case LineWidthCoupling.COUPLED:
                (
                    sextet_rvs.line_width1,
                    sextet_rvs.line_width2,
                    sextet_rvs.line_width3,
                    sextet_rvs.line_width4,
                    sextet_rvs.line_width5,
                    sextet_rvs.line_width6,
                ) = self._build_coupled_line_widths_rvs(
                    f"sx{k}_line_width123456",
                    [
                        f"sx{k}_line_width1",
                        f"sx{k}_line_width2",
                        f"sx{k}_line_width3",
                        f"sx{k}_line_width4",
                        f"sx{k}_line_width5",
                        f"sx{k}_line_width6",
                    ],
                    [
                        sextet.line_width1,
                        sextet.line_width2,
                        sextet.line_width3,
                        sextet.line_width4,
                        sextet.line_width5,
                        sextet.line_width6,
                    ],
                )
        return sextet_rvs


class BayesianModel(BayesianModelMixin, AAnalysis):
    """
    Full-Bayesian model for Mössbauer spectrum analysis.

    Attributes
    ----------
    specs : :any:`AnalysisSpecs`
        The specifications of the analysis model.
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "FULL_BAYES"
    results_type = AnalysisBayes

    def __init__(self, *args, **kwargs):
        """Initialize the Bayesian model."""
        super().__init__(*args, **kwargs)  # a single call is enough to invoke all parent constructors

        # additional attributes
        self.idata: az.InferenceData | None = None
        self.spectrum_rvs: SpectrumRVs | None = None

    def _clear_attributes(self):
        super()._clear_attributes()
        self.idata = None
        self.spectrum_rvs = None

    def _initialize(self):
        self.spectrum_rvs = self._build_spectrum_rvs(self._specs.spectrum)
        with self.model:
            y = spectrum_func(
                self._channels, spectrum=self.spectrum_rvs, spectroscope=self._specs.scope, geometry=self._geometry
            )
            likelihood = pm.Poisson("likelihood", mu=y, observed=self._counts)  # noqa: F841

    def _fit(self):
        self.idata = self._sample(chains=1, cores=1)

    def _create_results(self):
        post = az.extract(self.idata, group="posterior", combined=True)
        return AnalysisBayes(
            spectrum=self.spectrum_rvs.map(lambda x: BayesVar(posterior=post[x.name].values)),
            scope=self._specs.scope,
        )

    def predict(self, channels):
        """
        Predict the spectrum for the given channels.

        Parameters
        ----------
        channels : np.ndarray[int]
            The channels for which to predict the spectrum.

        Returns
        -------
        np.ndarray[float]
            The predicted spectrum.

        """
        return spectrum_func(
            channels,
            spectrum=self._results.spectrum.map(lambda x: x.value),
            spectroscope=self._specs.scope,
            geometry=self._geometry,
        )


class BayesianCalibrationModel(BayesianModelMixin, ACalibration):
    """
    Full-Bayesian model for Mössbauer spectrum calibration.

    Attributes
    ----------
    specs : :any:`CalibrationSpecs`
        The specifications of the calibration model.
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "FULL_BAYES"
    results_type = CalibrationBayes

    def __init__(self, *args, **kwargs):
        """Initialize the Bayesian model."""
        super().__init__(*args, **kwargs)  # a single call is enough to invoke all parent constructors

        # additional attributes
        self.idata: az.InferenceData | None = None
        self.scope_rvs: SpectroscopeRVs | None = None
        self.spectrum_rvs: SpectrumRVs | None = None

    def _clear_attributes(self):
        super()._clear_attributes()
        self.idata = None
        self.scope_rvs = None
        self.spectrum_rvs = None

    def _build_spectroscope_rvs(self, scope_props: SpectroscopeSpecs) -> SpectroscopeRVs:
        rvs = SpectroscopeRVs(scale=None, isomer_shift_ref=None)
        rvs.scale = self._build_scale_rvs(scope_props.scale)
        rvs.isomer_shift_ref = self._build_isomer_shift_rvs("isomer_shift_ref", scope_props.isomer_shift_ref)
        return rvs

    def _build_scale_rvs(self, scale: SpecsVar) -> Variable | pm.Distribution:
        with self.model:
            if scale.is_fixed:
                return pm.Deterministic("scale", pm.math.constant(scale.value))
            else:
                return pm.Normal("scale", mu=scale.value, sigma=scale.value * 0.2)

    def _initialize(self):
        self.scope_rvs = self._build_spectroscope_rvs(self._specs.scope)
        self.spectrum_rvs = self._build_spectrum_rvs(self._specs.spectrum)
        with self.model:
            y = spectrum_func(
                self._channels, spectrum=self.spectrum_rvs, spectroscope=self.scope_rvs, geometry=self._geometry
            )
            likelihood = pm.Poisson("likelihood", mu=y, observed=self._counts)  # noqa: F841

    def _fit(self):
        self.idata = self._sample(chains=1, cores=1)

    def _create_results(self):
        post = az.extract(self.idata, group="posterior", combined=True)
        return CalibrationBayes(
            spectrum=self.spectrum_rvs.map(lambda x: BayesVar(posterior=post[x.name].values)),
            scope=self.scope_rvs.map(lambda x: BayesVar(posterior=post[x.name].values)),
        )

    def predict(self, channels):
        """
        Predict the spectrum for the given channels.

        Parameters
        ----------
        channels : np.ndarray
            The channels for which to predict the spectrum

        Returns
        -------
        np.ndarray[float]
            The predicted spectrum

        """
        return spectrum_func(
            channels,
            spectrum=self._results.spectrum.map(lambda x: x.value),
            spectroscope=self._results.scope.map(lambda x: x.value),
            geometry=self._geometry,
        )


class MAPModel(BayesianModel):
    """
    Maximum a posteriori model for Mössbauer spectrum analysis.

    Attributes
    ----------
    specs : :any:`AnalysisSpecs`
        The specifications of the analysis model.
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "MAP"
    results_type = AnalysisPoint

    def _fit(self):
        self.map_estimate = self._find_map()

    def _create_results(self):
        return AnalysisPoint(
            spectrum=self.spectrum_rvs.map(lambda x: PointVar(value=self.map_estimate[x.name])),
            scope=self._specs.scope,
        )


class MAPCalibrationModel(BayesianCalibrationModel):
    """
    Maximum a posteriori model for Mössbauer spectrum calibration.

    Attributes
    ----------
    specs : :any:`CalibrationSpecs`
        The specifications of the calibration model.
    geometry : :any:`SpectroscopeGeometry`
        The geometry of the spectroscope.

    """

    name = "MAP"
    results_type = CalibrationPoint

    def _fit(self):
        self.map_estimate = self._find_map()

    def _create_results(self):
        return CalibrationPoint(
            spectrum=self.spectrum_rvs.map(lambda x: PointVar(value=self.map_estimate[x.name])),
            scope=self.scope_rvs.map(lambda x: PointVar(value=self.map_estimate[x.name])),
        )
