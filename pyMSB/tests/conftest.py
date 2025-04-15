import numpy as np

from pyMSB.models import (
    AnalysisSpecs,
    CalibrationSpecs,
    DoubletComputable,
    LineWidthCoupling,
    SextetComputable,
    SingletComputable,
    SpecsVar,
    SpectroscopeComputable,
    SpectrumComputable,
    SpectrumSpecs,
)

CHANNELS = np.arange(0, 1024, dtype=int)

SINGLET = SingletComputable(amplitude=10**4, isomer_shift=0, line_width1=10)
DOUBLET = DoubletComputable(amplitude=10**4, isomer_shift=0, quadrupole_split=150, line_width1=8, line_width2=8)
SEXTET = SextetComputable(
    amplitude=10**4,
    isomer_shift=0,
    quadrupole_split=0,
    ratio13=3,
    ratio23=2,
    magnetic_field=30,
    line_width1=8,
    line_width2=8,
    line_width3=8,
    line_width4=8,
    line_width5=8,
    line_width6=8,
)
SPECTRUM = SpectrumComputable(background=10**5, singlets=[SINGLET], doublets=[DOUBLET], sextets=[SEXTET])
SPECTROSCOPE = SpectroscopeComputable(scale=40, isomer_shift_ref=512)

# Computables mapped to Specs
SINGLET_SPECS = SINGLET.map(lambda x: SpecsVar(value=x, is_fixed=False))
DOUBLET_SPECS = DOUBLET.map(
    lambda x: SpecsVar(value=x, is_fixed=False), line_width_coupling=LineWidthCoupling.UNCOUPLED
)
SEXTET_SPECS = SEXTET.map(lambda x: SpecsVar(value=x, is_fixed=False), line_width_coupling=LineWidthCoupling.UNCOUPLED)
SEXTET_SPECS.isomer_shift.is_fixed = True
SEXTET_SPECS.magnetic_field.is_fixed = True

CALIBRATION_SPECTRUM_SPECS = SpectrumSpecs(background=SpecsVar(value=10**5, is_fixed=False), sextets=[SEXTET_SPECS])
SPECTROSCOPE_SPECS = SPECTROSCOPE.map(lambda x: SpecsVar(value=x, is_fixed=False))
CALIBRATION_SPECS = CalibrationSpecs(spectrum=CALIBRATION_SPECTRUM_SPECS, scope=SPECTROSCOPE_SPECS)

ANALYSIS_SPECTRUM_SPECS = SpectrumSpecs(
    background=SpecsVar(value=10**5, is_fixed=False), singlets=[SINGLET_SPECS], doublets=[DOUBLET_SPECS]
)
ANALYSIS_SPECS = AnalysisSpecs(spectrum=ANALYSIS_SPECTRUM_SPECS, scope=SPECTROSCOPE)

ALL_SPECTRUM_SPECS = SpectrumSpecs(
    background=SpecsVar(value=10**5, is_fixed=False),
    singlets=[SINGLET_SPECS],
    doublets=[DOUBLET_SPECS],
    sextets=[SEXTET_SPECS],
)
