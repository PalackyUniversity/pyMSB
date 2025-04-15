import matplotlib.pyplot as plt
import numpy as np

from pyMSB import (
    BayesianCalibrationModel,
    BayesianModel,
    LSQCalibrationModel,
    LSQModel,
    MAPCalibrationModel,
    MAPModel,
)
from pyMSB.common import generate_spectrum, spectrum_func
from pyMSB.models import (
    AnalysisSpecs,
    CalibrationSpecs,
    SextetSpecs,
    SingletSpecs,
    SpecsVar,
    SpectroscopeComputable,
    SpectroscopeGeometry,
    SpectroscopeSpecs,
    SpectrumSpecs,
)

# list of models - Pycharm kept removing unused imports
models = [BayesianCalibrationModel, LSQCalibrationModel, MAPCalibrationModel, BayesianModel, LSQModel, MAPModel]


### SELECT MODELS to test ###

# CALIBRATION_MODEL = BayesianCalibrationModel
# ANALYSIS_MODEL = BayesianModel

CALIBRATION_MODEL = LSQCalibrationModel
ANALYSIS_MODEL = LSQModel

# CALIBRATION_MODEL = MAPCalibrationModel
# ANALYSIS_MODEL = MAPModel


CHANNELS = np.arange(0, 1024, dtype=int)

sextet_specs = SextetSpecs(
    amplitude=SpecsVar(value=10**4),
    isomer_shift=SpecsVar(value=0, is_fixed=True),
    quadrupole_split=SpecsVar(value=0),
    ratio13=SpecsVar(value=3),
    ratio23=SpecsVar(value=2),
    magnetic_field=SpecsVar(value=30, is_fixed=True),
    line_width1=SpecsVar(value=8),
    line_width2=SpecsVar(value=8),
    line_width3=SpecsVar(value=8),
    line_width4=SpecsVar(value=8),
    line_width5=SpecsVar(value=8),
    line_width6=SpecsVar(value=8),
)

calibration_spectrum_specs = SpectrumSpecs(background=SpecsVar(value=10**5), sextets=[sextet_specs])
spectroscope_specs = SpectroscopeSpecs(scale=SpecsVar(value=40), isomer_shift_ref=SpecsVar(value=512))

calibration_specs = CalibrationSpecs(spectrum=calibration_spectrum_specs, scope=spectroscope_specs)

spectroscope_generic = SpectroscopeComputable(scale=40, isomer_shift_ref=512)

calibration_func = spectrum_func(
    channels=CHANNELS,
    spectrum=calibration_spectrum_specs.map(lambda x: x.value),
    spectroscope=spectroscope_generic,
    geometry=SpectroscopeGeometry.TRANSMISSION,
)

calibration_counts = generate_spectrum(
    channels=CHANNELS,
    spectrum=calibration_spectrum_specs.map(lambda x: x.value),
    spectroscope=spectroscope_generic,
    geometry=SpectroscopeGeometry.TRANSMISSION,
    seed=0,
)

calibration_model = CALIBRATION_MODEL(specs=calibration_specs, geometry=SpectroscopeGeometry.TRANSMISSION)
calibration_model.fit(channels=CHANNELS, counts=calibration_counts)

calibration_result = calibration_model.results
prediction = calibration_model.predict(CHANNELS)

plt.plot(CHANNELS, calibration_func, label="calib_func")
plt.plot(CHANNELS, calibration_counts, "k+", label="calib_counts")
plt.plot(CHANNELS, prediction, label="estimate")
plt.legend()
plt.show()


# ANALYSIS

singlet_specs = SingletSpecs(
    amplitude=SpecsVar(value=10**4), isomer_shift=SpecsVar(value=50), line_width1=SpecsVar(value=10)
)

anal_spectrum_specs = SpectrumSpecs(background=SpecsVar(value=10**5), singlets=[singlet_specs])
spectroscope_from_cal = calibration_result.scope.map(lambda x: x.value)

anal_specs = AnalysisSpecs(spectrum=anal_spectrum_specs, scope=spectroscope_from_cal)

anal_func = spectrum_func(
    channels=CHANNELS,
    spectrum=anal_spectrum_specs.map(lambda x: x.value),
    spectroscope=spectroscope_from_cal,
    geometry=SpectroscopeGeometry.TRANSMISSION,
)

anal_counts = generate_spectrum(
    channels=CHANNELS,
    spectrum=anal_spectrum_specs.map(lambda x: x.value),
    spectroscope=spectroscope_from_cal,
    geometry=SpectroscopeGeometry.TRANSMISSION,
    seed=0,
)

anal_model = ANALYSIS_MODEL(specs=anal_specs, geometry=SpectroscopeGeometry.TRANSMISSION)
anal_model.fit(channels=CHANNELS, counts=anal_counts)

anal_result = anal_model.results
prediction = anal_model.predict(CHANNELS)

plt.plot(CHANNELS, anal_func, label="anal_func")
plt.plot(CHANNELS, anal_counts, "k+", label="anal_counts")
plt.plot(CHANNELS, prediction, label="estimate")
plt.legend()
plt.show()
