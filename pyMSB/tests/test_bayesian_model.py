from copy import deepcopy

import pymc as pm
import pytest

from pyMSB.computation_models import BayesianCalibrationModel, BayesianModel
from pyMSB.exceptions import ValidationExc
from pyMSB.models import AnalysisSpecs, LineWidthCoupling, SpecsVar, SpectroscopeGeometry
from pyMSB.tests.conftest import (
    ALL_SPECTRUM_SPECS,
    CALIBRATION_SPECS,
    DOUBLET_SPECS,
    SEXTET_SPECS,
    SINGLET_SPECS,
    SPECTROSCOPE,
    SPECTROSCOPE_SPECS,
)


@pytest.fixture
def bayes_model():
    bayes_model = BayesianModel(
        specs=AnalysisSpecs(spectrum=ALL_SPECTRUM_SPECS, scope=SPECTROSCOPE), geometry=SpectroscopeGeometry.REFLECTION
    )
    bayes_model._channels = [0, 1, 2, 3, 4]
    bayes_model._counts = [0, 0, 1, 0, 0]
    return bayes_model


@pytest.fixture
def calibration_model():
    calibration_model = BayesianCalibrationModel(specs=CALIBRATION_SPECS, geometry=SpectroscopeGeometry.REFLECTION)
    calibration_model._channels = [0, 1, 2, 3, 4]
    calibration_model._counts = [0, 0, 1, 0, 0]
    return calibration_model


def assert_rv_in_model(bayes_model, rv_to_test, corresponding_spec):
    """
    Assert that the RV is in the model.

    If the corresponding spec is fixed, the RV should be in the deterministics.
    Otherwise, it should be in the basic_RVs.
    """
    if not corresponding_spec.is_fixed:
        assert rv_to_test in bayes_model.model.basic_RVs
    else:
        assert rv_to_test in bayes_model.model.deterministics


def test_build_spectrum_rvs(bayes_model):
    rvs = bayes_model._build_spectrum_rvs(ALL_SPECTRUM_SPECS)
    assert bayes_model.model.deterministics
    assert bayes_model.model.basic_RVs
    assert rvs.background in bayes_model.model.deterministics
    assert rvs.singlets[0].amplitude in bayes_model.model.deterministics
    assert rvs.doublets[0].amplitude in bayes_model.model.deterministics
    assert rvs.sextets[0].amplitude in bayes_model.model.deterministics


def test_build_model(bayes_model):
    bayes_model._initialize()
    assert bayes_model.model.deterministics
    assert bayes_model.model.basic_RVs
    assert bayes_model.model.observed_RVs
    assert "likelihood" in (rv.name for rv in bayes_model.model.observed_RVs)
    assert bayes_model.spectrum_rvs
    assert bayes_model.spectrum_rvs.background in bayes_model.model.deterministics


def test_build_background_rvs_with_fixed_value(bayes_model):
    bckg = SpecsVar(value=10, is_fixed=True)
    background_rv = bayes_model._build_background_rvs(bckg)
    assert f"{background_rv.name}_coeff" not in bayes_model.model.named_vars
    assert background_rv.eval() == 10


def test_build_background_rvs_with_non_fixed_value(bayes_model):
    bckg = SpecsVar(value=10, is_fixed=False)
    background_rv = bayes_model._build_background_rvs(bckg)
    assert f"{background_rv.name}_coeff" in bayes_model.model.named_vars
    assert background_rv in bayes_model.model.deterministics


def test_build_amplitude_rvs_with_fixed_value(bayes_model):
    amplitude = SpecsVar(value=10, is_fixed=True)
    amplitude_rv = bayes_model._build_amplitude_rvs("amplitude", amplitude)
    assert f"{amplitude_rv.name}_coeff" not in bayes_model.model.named_vars
    assert amplitude_rv.eval() == 10


def test_build_amplitude_rvs_with_non_fixed_value(bayes_model):
    amplitude = SpecsVar(value=10, is_fixed=False)
    amplitude_rv = bayes_model._build_amplitude_rvs("amplitude", amplitude)
    assert f"{amplitude_rv.name}_coeff" in bayes_model.model.named_vars
    assert amplitude_rv in bayes_model.model.deterministics


def test_build_isomer_shift_rvs_with_fixed_value(bayes_model):
    isomer_shift = SpecsVar(value=10, is_fixed=True)
    isomer_shift_rv = bayes_model._build_isomer_shift_rvs("isomer_shift", isomer_shift)
    assert isomer_shift_rv in bayes_model.model.deterministics
    assert isomer_shift_rv.eval() == 10


def test_build_isomer_shift_rvs_with_non_fixed_value(bayes_model):
    isomer_shift = SpecsVar(value=10, is_fixed=False)
    isomer_shift_rv = bayes_model._build_isomer_shift_rvs("isomer_shift", isomer_shift)
    assert isinstance(isomer_shift_rv.owner.op, pm.Normal)
    assert isomer_shift_rv in bayes_model.model.basic_RVs


def test_build_line_width_rvs_with_fixed_value(bayes_model):
    line_width = SpecsVar(value=10, is_fixed=True)
    line_width_rv = bayes_model._build_line_width_rvs("line_width", line_width)
    assert line_width_rv in bayes_model.model.deterministics
    assert line_width_rv.eval() == 10


def test_build_line_width_rv_with_non_fixed_value(bayes_model):
    line_width = SpecsVar(value=10, is_fixed=False)
    line_width_rv = bayes_model._build_line_width_rvs("line_width", line_width)
    assert isinstance(line_width_rv.owner.op, pm.LogNormal)
    assert line_width_rv in bayes_model.model.basic_RVs


def test_build_coupled_line_widths_rvs_with_fixed_values(bayes_model):
    line_widths = [SpecsVar(value=10, is_fixed=True), SpecsVar(value=20, is_fixed=True)]
    line_width_rvs = bayes_model._build_coupled_line_widths_rvs(
        "line_widths12", ["line_width1", "line_width2"], line_widths
    )
    assert "line_widths12" in map(lambda rv: rv.name, bayes_model.model.deterministics)
    for rv in line_width_rvs:
        assert rv in bayes_model.model.deterministics


def test_build_coupled_line_widths_rvs_with_conflicting_fixation(bayes_model):
    line_widths = [SpecsVar(value=10, is_fixed=False), SpecsVar(value=20, is_fixed=True)]
    with pytest.raises(ValidationExc):
        bayes_model._build_coupled_line_widths_rvs("line_widths12", ["line_width1", "line_width2"], line_widths)


def test_build_coupled_line_widths_rvs_with_non_fixed_values(bayes_model):
    line_widths = [SpecsVar(value=10, is_fixed=False), SpecsVar(value=20, is_fixed=False)]
    line_width_rvs = bayes_model._build_coupled_line_widths_rvs(
        "line_widths12", ["line_width1", "line_width2"], line_widths
    )
    assert "line_widths12" in bayes_model.model.named_vars
    for rv in line_width_rvs:
        assert rv in bayes_model.model.deterministics


def test_build_quadrupole_split_rvs_with_fixed_value(bayes_model):
    quadrupole_split = SpecsVar(value=10, is_fixed=True)
    quadrupole_split_rv = bayes_model._build_quadrupole_split_rvs("quadrupole_split", quadrupole_split)
    assert quadrupole_split_rv in bayes_model.model.deterministics
    assert quadrupole_split_rv.eval() == 10


def test_build_quadrupole_split_rvs_with_non_fixed_value(bayes_model):
    quadrupole_split = SpecsVar(value=10, is_fixed=False)
    quadrupole_split_rv = bayes_model._build_quadrupole_split_rvs("quadrupole_split", quadrupole_split)
    assert isinstance(quadrupole_split_rv.owner.op, pm.Normal)
    assert quadrupole_split_rv in bayes_model.model.basic_RVs


def test_build_ratio_rvs_with_fixed_value(bayes_model):
    ratio = SpecsVar(value=10, is_fixed=True)
    ratio_rv = bayes_model._build_ratio_rvs("ratio", ratio)
    assert ratio_rv in bayes_model.model.deterministics
    assert ratio_rv.eval() == 10


def test_build_ratio_rvs_with_non_fixed_value(bayes_model):
    ratio = SpecsVar(value=10, is_fixed=False)
    ratio_rv = bayes_model._build_ratio_rvs("ratio", ratio)
    assert isinstance(ratio_rv.owner.op, pm.Normal)
    assert ratio_rv in bayes_model.model.basic_RVs


def test_build_magnetic_field_rvs_with_fixed_value(bayes_model):
    magnetic_field = SpecsVar(value=10, is_fixed=True)
    magnetic_field_rv = bayes_model._build_magnetic_field_rvs("magnetic_field", magnetic_field)
    assert magnetic_field_rv in bayes_model.model.deterministics
    assert magnetic_field_rv.eval() == 10


def test_build_magnetic_field_rvs_with_non_fixed_value(bayes_model):
    magnetic_field = SpecsVar(value=10, is_fixed=False)
    magnetic_field_rv = bayes_model._build_magnetic_field_rvs("magnetic_field", magnetic_field)
    assert isinstance(magnetic_field_rv.owner.op, pm.Normal)
    assert magnetic_field_rv in bayes_model.model.basic_RVs


def test_singlet_rvs(bayes_model):
    singlet_rvs = bayes_model._build_singlet_rvs(SINGLET_SPECS, k=1)
    assert singlet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, singlet_rvs.isomer_shift, SINGLET_SPECS.isomer_shift)
    assert_rv_in_model(bayes_model, singlet_rvs.line_width1, SINGLET_SPECS.line_width1)


def test_build_doublet_rvs_uncoupled(bayes_model):
    doublet_specs = deepcopy(DOUBLET_SPECS)
    doublet_rvs = bayes_model._build_doublet_rvs(doublet_specs, k=1)
    assert doublet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, doublet_rvs.isomer_shift, doublet_specs.isomer_shift)
    assert_rv_in_model(bayes_model, doublet_rvs.quadrupole_split, doublet_specs.quadrupole_split)
    assert_rv_in_model(bayes_model, doublet_rvs.line_width1, doublet_specs.line_width1)
    assert_rv_in_model(bayes_model, doublet_rvs.line_width2, doublet_specs.line_width2)


def test_build_doublet_rvs_coupled(bayes_model):
    doublet_specs = deepcopy(DOUBLET_SPECS)
    doublet_specs.line_width_coupling = LineWidthCoupling.COUPLED
    doublet_rvs = bayes_model._build_doublet_rvs(doublet_specs, k=2)
    assert doublet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, doublet_rvs.isomer_shift, doublet_specs.isomer_shift)
    assert_rv_in_model(bayes_model, doublet_rvs.quadrupole_split, doublet_specs.quadrupole_split)
    assert doublet_rvs.line_width1 in bayes_model.model.deterministics
    assert doublet_rvs.line_width2 in bayes_model.model.deterministics


def test_build_sextet_rvs_uncoupled(bayes_model):
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_rvs = bayes_model._build_sextet_rvs(sextet_specs, k=1)
    assert sextet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, sextet_rvs.isomer_shift, sextet_specs.isomer_shift)
    assert_rv_in_model(bayes_model, sextet_rvs.quadrupole_split, sextet_specs.quadrupole_split)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio13, sextet_specs.ratio13)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio23, sextet_specs.ratio23)
    assert_rv_in_model(bayes_model, sextet_rvs.magnetic_field, sextet_specs.magnetic_field)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width1, sextet_specs.line_width1)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width2, sextet_specs.line_width2)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width3, sextet_specs.line_width3)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width4, sextet_specs.line_width4)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width5, sextet_specs.line_width5)
    assert_rv_in_model(bayes_model, sextet_rvs.line_width6, sextet_specs.line_width6)


def test_build_sextet_rvs_coupled(bayes_model):
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_specs.line_width_coupling = LineWidthCoupling.COUPLED
    sextet_rvs = bayes_model._build_sextet_rvs(sextet_specs, k=2)
    assert sextet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, sextet_rvs.isomer_shift, sextet_specs.isomer_shift)
    assert_rv_in_model(bayes_model, sextet_rvs.quadrupole_split, sextet_specs.quadrupole_split)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio13, sextet_specs.ratio13)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio23, sextet_specs.ratio23)
    assert_rv_in_model(bayes_model, sextet_rvs.magnetic_field, sextet_specs.magnetic_field)
    assert "sx2_line_width123456" in bayes_model.model.named_vars
    assert sextet_rvs.line_width1 in bayes_model.model.deterministics
    assert sextet_rvs.line_width2 in bayes_model.model.deterministics
    assert sextet_rvs.line_width3 in bayes_model.model.deterministics
    assert sextet_rvs.line_width4 in bayes_model.model.deterministics
    assert sextet_rvs.line_width5 in bayes_model.model.deterministics
    assert sextet_rvs.line_width6 in bayes_model.model.deterministics


def test_build_sextet_rvs_coupled_pairs(bayes_model):
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_specs.line_width_coupling = LineWidthCoupling.COUPLED_PAIRS
    sextet_rvs = bayes_model._build_sextet_rvs(sextet_specs, k=2)
    assert sextet_rvs.amplitude in bayes_model.model.deterministics
    assert_rv_in_model(bayes_model, sextet_rvs.isomer_shift, sextet_specs.isomer_shift)
    assert_rv_in_model(bayes_model, sextet_rvs.quadrupole_split, sextet_specs.quadrupole_split)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio13, sextet_specs.ratio13)
    assert_rv_in_model(bayes_model, sextet_rvs.ratio23, sextet_specs.ratio23)
    assert_rv_in_model(bayes_model, sextet_rvs.magnetic_field, sextet_specs.magnetic_field)
    assert "sx2_line_width16" in bayes_model.model.named_vars
    assert "sx2_line_width25" in bayes_model.model.named_vars
    assert "sx2_line_width34" in bayes_model.model.named_vars
    assert sextet_rvs.line_width1 in bayes_model.model.deterministics
    assert sextet_rvs.line_width2 in bayes_model.model.deterministics
    assert sextet_rvs.line_width3 in bayes_model.model.deterministics
    assert sextet_rvs.line_width4 in bayes_model.model.deterministics
    assert sextet_rvs.line_width5 in bayes_model.model.deterministics
    assert sextet_rvs.line_width6 in bayes_model.model.deterministics


# ------------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------------


def test_build_spectroscope_rvs(calibration_model):
    rvs = calibration_model._build_spectroscope_rvs(SPECTROSCOPE_SPECS)
    assert_rv_in_model(calibration_model, rvs.scale, SPECTROSCOPE_SPECS.scale)
    assert_rv_in_model(calibration_model, rvs.isomer_shift_ref, SPECTROSCOPE_SPECS.isomer_shift_ref)


def test_build_model_calibration(calibration_model):
    calibration_model._initialize()
    assert calibration_model.model.basic_RVs
    assert "likelihood" in (rv.name for rv in calibration_model.model.observed_RVs)
    assert calibration_model.spectrum_rvs
    assert calibration_model.spectrum_rvs.background in calibration_model.model.deterministics
    assert calibration_model.scope_rvs
    assert_rv_in_model(calibration_model, calibration_model.scope_rvs.scale, SPECTROSCOPE_SPECS.scale)


def test_build_scale_rvs_with_fixed_value(calibration_model):
    scale = SpecsVar(value=10, is_fixed=True)
    scale_rv = calibration_model._build_scale_rvs(scale)
    assert scale_rv in calibration_model.model.deterministics
    assert scale_rv.eval() == 10


def test_build_scale_rvs_with_non_fixed_value(calibration_model):
    scale = SpecsVar(value=10, is_fixed=False)
    scale_rv = calibration_model._build_scale_rvs(scale)
    assert isinstance(scale_rv.owner.op, pm.Normal)
    assert_rv_in_model(calibration_model, scale_rv, scale)
