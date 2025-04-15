from copy import deepcopy

import pytest

from pyMSB.computation_models import LSQCalibrationModel, LSQModel
from pyMSB.models import LineWidthCoupling, SpecsVar, SpectroscopeGeometry
from pyMSB.tests.conftest import (
    ALL_SPECTRUM_SPECS,
    ANALYSIS_SPECS,
    CALIBRATION_SPECS,
    DOUBLET_SPECS,
    SEXTET_SPECS,
    SINGLET_SPECS,
    SPECTROSCOPE,
    SPECTROSCOPE_SPECS,
)


@pytest.fixture
def lsq_model():
    return LSQModel(specs=deepcopy(ANALYSIS_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


@pytest.fixture
def lsq_calibration_model():
    return LSQCalibrationModel(specs=deepcopy(CALIBRATION_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


# ------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------


def test_flatten_doublet_fit_params():
    """Test if doublet fit parameters are flattened correctly."""
    doublet_specs = deepcopy(DOUBLET_SPECS)
    true = []
    true += [doublet_specs.amplitude.value] if not doublet_specs.amplitude.is_fixed else []
    true += [doublet_specs.isomer_shift.value] if not doublet_specs.isomer_shift.is_fixed else []
    true += [doublet_specs.quadrupole_split.value] if not doublet_specs.quadrupole_split.is_fixed else []
    true += [doublet_specs.line_width1.value] if not doublet_specs.line_width1.is_fixed else []
    true += [doublet_specs.line_width2.value] if not doublet_specs.line_width2.is_fixed else []
    flat_params = LSQModel._flatten_doublet_fit_params(doublet_specs)
    # TODO: Fix
    assert flat_params == true


def test_flatten_doublet_fit_params_locked():
    """Test if doublet fit parameters are flattened correctly - locked case."""
    doublet_specs = deepcopy(DOUBLET_SPECS)
    doublet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    doublet_specs.isomer_shift.is_fixed = True
    doublet_specs.quadrupole_split.is_fixed = True
    doublet_specs.line_width1.is_fixed = True
    doublet_specs.line_width2.is_fixed = True
    doublet_specs.line_width_coupling = LineWidthCoupling.UNCOUPLED
    flat_params = LSQModel._flatten_doublet_fit_params(doublet_specs)
    assert flat_params == []


def test_flatten_doublet_fit_params_coupled():
    """Test if doublet fit parameters are flattened correctly - coupled case."""
    doublet_specs = deepcopy(DOUBLET_SPECS)
    doublet_specs.line_width_coupling = LineWidthCoupling.COUPLED
    doublet_specs_coupled_pairs = deepcopy(DOUBLET_SPECS)
    doublet_specs_coupled_pairs.line_width_coupling = LineWidthCoupling.COUPLED_PAIRS
    true = []
    true += [doublet_specs.amplitude.value] if not doublet_specs.amplitude.is_fixed else []
    true += [doublet_specs.isomer_shift.value] if not doublet_specs.isomer_shift.is_fixed else []
    true += [doublet_specs.quadrupole_split.value] if not doublet_specs.quadrupole_split.is_fixed else []
    true += [doublet_specs.line_width1.value] if not doublet_specs.line_width1.is_fixed else []
    flat_coupled = LSQModel._flatten_doublet_fit_params(doublet_specs)
    flat_coupled_pairs = LSQModel._flatten_doublet_fit_params(doublet_specs_coupled_pairs)
    assert flat_coupled == true
    assert flat_coupled_pairs == true


def test_flatten_sextet_fit_params():
    """Test if sextet fit parameters are flattened correctly."""
    sextet_specs = deepcopy(SEXTET_SPECS)
    true = []
    true += [sextet_specs.amplitude.value] if not sextet_specs.amplitude.is_fixed else []
    true += [sextet_specs.isomer_shift.value] if not sextet_specs.isomer_shift.is_fixed else []
    true += [sextet_specs.quadrupole_split.value] if not sextet_specs.quadrupole_split.is_fixed else []
    true += [sextet_specs.ratio13.value] if not sextet_specs.ratio13.is_fixed else []
    true += [sextet_specs.ratio23.value] if not sextet_specs.ratio23.is_fixed else []
    true += [sextet_specs.magnetic_field.value] if not sextet_specs.magnetic_field.is_fixed else []
    true += [sextet_specs.line_width1.value] if not sextet_specs.line_width1.is_fixed else []
    true += [sextet_specs.line_width2.value] if not sextet_specs.line_width2.is_fixed else []
    true += [sextet_specs.line_width3.value] if not sextet_specs.line_width3.is_fixed else []
    true += [sextet_specs.line_width4.value] if not sextet_specs.line_width4.is_fixed else []
    true += [sextet_specs.line_width5.value] if not sextet_specs.line_width5.is_fixed else []
    true += [sextet_specs.line_width6.value] if not sextet_specs.line_width6.is_fixed else []
    flat_params = LSQModel._flatten_sextet_fit_params(sextet_specs)
    # TODO: Fix
    assert flat_params == true


def test_flatten_sextet_fit_params_locked():
    """Test if sextet fit parameters are flattened correctly - locked case."""
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    sextet_specs.isomer_shift.is_fixed = True
    sextet_specs.quadrupole_split.is_fixed = True
    sextet_specs.ratio13.is_fixed = True
    sextet_specs.ratio23.is_fixed = True
    sextet_specs.magnetic_field.is_fixed = True
    sextet_specs.line_width1.is_fixed = True
    sextet_specs.line_width2.is_fixed = True
    sextet_specs.line_width3.is_fixed = True
    sextet_specs.line_width4.is_fixed = True
    sextet_specs.line_width5.is_fixed = True
    sextet_specs.line_width6.is_fixed = True
    sextet_specs.line_width_coupling = LineWidthCoupling.UNCOUPLED
    flat_params = LSQModel._flatten_sextet_fit_params(sextet_specs)
    assert flat_params == []


def test_flatten_sextet_fit_params_coupled():
    """Test if sextet fit parameters are flattened correctly - coupled case."""
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_specs.line_width_coupling = LineWidthCoupling.COUPLED
    true = []
    true += [sextet_specs.amplitude.value] if not sextet_specs.amplitude.is_fixed else []
    true += [sextet_specs.isomer_shift.value] if not sextet_specs.isomer_shift.is_fixed else []
    true += [sextet_specs.quadrupole_split.value] if not sextet_specs.quadrupole_split.is_fixed else []
    true += [sextet_specs.ratio13.value] if not sextet_specs.ratio13.is_fixed else []
    true += [sextet_specs.ratio23.value] if not sextet_specs.ratio23.is_fixed else []
    true += [sextet_specs.magnetic_field.value] if not sextet_specs.magnetic_field.is_fixed else []
    true += [sextet_specs.line_width1.value] if not sextet_specs.line_width1.is_fixed else []
    flat_coupled = LSQModel._flatten_sextet_fit_params(sextet_specs)
    assert flat_coupled == true


def test_flatten_sextet_fit_params_coupled_pairs():
    """Test if sextet fit parameters are flattened correctly - coupled pairs case."""
    sextet = deepcopy(SEXTET_SPECS)
    sextet.line_width_coupling = LineWidthCoupling.COUPLED_PAIRS
    true = []
    true += [sextet.amplitude.value] if not sextet.amplitude.is_fixed else []
    true += [sextet.isomer_shift.value] if not sextet.isomer_shift.is_fixed else []
    true += [sextet.quadrupole_split.value] if not sextet.quadrupole_split.is_fixed else []
    true += [sextet.ratio13.value] if not sextet.ratio13.is_fixed else []
    true += [sextet.ratio23.value] if not sextet.ratio23.is_fixed else []
    true += [sextet.magnetic_field.value] if not sextet.magnetic_field.is_fixed else []
    true += [sextet.line_width1.value] if not sextet.line_width1.is_fixed else []
    true += [sextet.line_width2.value] if not sextet.line_width2.is_fixed else []
    true += [sextet.line_width3.value] if not sextet.line_width3.is_fixed else []
    flat_coupled_pairs = LSQModel._flatten_sextet_fit_params(sextet)
    assert flat_coupled_pairs == true


def test_flatten_singlet_fit_params(lsq_model):
    """Test if singlet fit parameters are flattened correctly."""
    singlet_specs = deepcopy(SINGLET_SPECS)
    true = []
    true += [singlet_specs.amplitude.value] if not singlet_specs.amplitude.is_fixed else []
    true += [singlet_specs.isomer_shift.value] if not singlet_specs.isomer_shift.is_fixed else []
    true += [singlet_specs.line_width1.value] if not singlet_specs.line_width1.is_fixed else []
    flat_params = lsq_model._flatten_fit_params(singlet_specs)
    assert flat_params == true


def test_flatten_singlet_fit_params_locked(lsq_model):
    """Test if singlet fit parameters are flattened correctly - locked case."""
    singlet_specs = deepcopy(SINGLET_SPECS)
    singlet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    singlet_specs.isomer_shift.is_fixed = True
    singlet_specs.line_width1.is_fixed = True
    flat_params = lsq_model._flatten_fit_params(singlet_specs)
    assert flat_params == []


def test_flatted_spectrum_fit_params(lsq_model):
    """Test if spectrum fit parameters are flattened correctly."""
    true = []
    true += [ALL_SPECTRUM_SPECS.background.value] if not ALL_SPECTRUM_SPECS.background.is_fixed else []
    for singlet_specs in ALL_SPECTRUM_SPECS.singlets:
        true += [singlet_specs.amplitude.value] if not singlet_specs.amplitude.is_fixed else []
        true += [singlet_specs.isomer_shift.value] if not singlet_specs.isomer_shift.is_fixed else []
        true += [singlet_specs.line_width1.value] if not singlet_specs.line_width1.is_fixed else []
    for doublet_specs in ALL_SPECTRUM_SPECS.doublets:
        true += [doublet_specs.amplitude.value] if not doublet_specs.amplitude.is_fixed else []
        true += [doublet_specs.isomer_shift.value] if not doublet_specs.isomer_shift.is_fixed else []
        true += [doublet_specs.quadrupole_split.value] if not doublet_specs.quadrupole_split.is_fixed else []
        true += [doublet_specs.line_width1.value] if not doublet_specs.line_width1.is_fixed else []
        true += [doublet_specs.line_width2.value] if not doublet_specs.line_width2.is_fixed else []
    for sextet_specs in ALL_SPECTRUM_SPECS.sextets:
        true += [sextet_specs.amplitude.value] if not sextet_specs.amplitude.is_fixed else []
        true += [sextet_specs.isomer_shift.value] if not sextet_specs.isomer_shift.is_fixed else []
        true += [sextet_specs.quadrupole_split.value] if not sextet_specs.quadrupole_split.is_fixed else []
        true += [sextet_specs.ratio13.value] if not sextet_specs.ratio13.is_fixed else []
        true += [sextet_specs.ratio23.value] if not sextet_specs.ratio23.is_fixed else []
        true += [sextet_specs.magnetic_field.value] if not sextet_specs.magnetic_field.is_fixed else []
        true += [sextet_specs.line_width1.value] if not sextet_specs.line_width1.is_fixed else []
        true += [sextet_specs.line_width2.value] if not sextet_specs.line_width2.is_fixed else []
        true += [sextet_specs.line_width3.value] if not sextet_specs.line_width3.is_fixed else []
        true += [sextet_specs.line_width4.value] if not sextet_specs.line_width4.is_fixed else []
        true += [sextet_specs.line_width5.value] if not sextet_specs.line_width5.is_fixed else []
        true += [sextet_specs.line_width6.value] if not sextet_specs.line_width6.is_fixed else []
    flat_params = lsq_model._flatten_fit_params(ALL_SPECTRUM_SPECS)
    assert flat_params == true


def test_rebuild_doublet_from_fit_params():
    """Test if doublet is rebuilt correctly from flattened fit parameters."""
    doublet_specs = deepcopy(DOUBLET_SPECS)
    flat_params = LSQModel._flatten_doublet_fit_params(doublet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    LSQModel._rebuild_doublet_from_fit_params(doublet_specs, iterator)
    assert doublet_specs == DOUBLET_SPECS


def test_rebuild_doublet_from_fit_params_locked():
    """Test if doublet is rebuilt correctly from flattened fit parameters - locked case."""
    doublet_specs = deepcopy(DOUBLET_SPECS)
    doublet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    doublet_specs.isomer_shift.is_fixed = True
    doublet_specs.quadrupole_split.is_fixed = True
    doublet_specs.line_width1.is_fixed = True
    doublet_specs.line_width2.is_fixed = True
    doublet_orig = deepcopy(doublet_specs)
    flat_params = LSQModel._flatten_doublet_fit_params(doublet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    LSQModel._rebuild_doublet_from_fit_params(doublet_specs, iterator)
    assert doublet_specs == doublet_orig


def test_rebuild_doublet_from_fit_params_coupled():
    """Test if doublet is rebuilt correctly from flattened fit parameters - coupled case."""
    doublet_specs_coupled = deepcopy(DOUBLET_SPECS)
    doublet_specs_coupled.line_width_coupling = LineWidthCoupling.COUPLED
    doublet_specs_coupled_orig = deepcopy(doublet_specs_coupled)
    coupled_flat = LSQModel._flatten_doublet_fit_params(doublet_specs_coupled)
    coupled_iterator = iter(list(map(lambda param: SpecsVar(param), coupled_flat)))
    LSQModel._rebuild_doublet_from_fit_params(doublet_specs_coupled, coupled_iterator)
    assert doublet_specs_coupled_orig == doublet_specs_coupled


def test_rebuild_doublet_from_fit_params_coupled_pairs():
    """Test if doublet is rebuilt correctly from flattened fit parameters - coupled pairs case."""
    doublet_specs_coupled_pairs = deepcopy(DOUBLET_SPECS)
    doublet_specs_coupled_pairs.line_width_coupling = LineWidthCoupling.COUPLED_PAIRS
    doublet_specs_coupled_pairs_orig = deepcopy(doublet_specs_coupled_pairs)
    coupled_pairs_flat = LSQModel._flatten_doublet_fit_params(doublet_specs_coupled_pairs)
    coupled_pairs_iterator = iter(list(map(lambda param: SpecsVar(param), coupled_pairs_flat)))
    LSQModel._rebuild_doublet_from_fit_params(doublet_specs_coupled_pairs, coupled_pairs_iterator)
    assert doublet_specs_coupled_pairs_orig == doublet_specs_coupled_pairs


def test_rebuild_sextet_from_fit_params():
    """Test if sextet is rebuilt correctly from flattened fit parameters."""
    sextet_specs = deepcopy(SEXTET_SPECS)
    flat_params = LSQModel._flatten_sextet_fit_params(sextet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    LSQModel._rebuild_sextet_from_fit_params(sextet_specs, iterator)
    assert sextet_specs == SEXTET_SPECS


def test_rebuild_sextet_from_fit_params_locked():
    """Test if sextet is rebuilt correctly from flattened fit parameters - locked case."""
    sextet_specs = deepcopy(SEXTET_SPECS)
    sextet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    sextet_specs.isomer_shift.is_fixed = True
    sextet_specs.quadrupole_split.is_fixed = True
    sextet_specs.ratio13.is_fixed = True
    sextet_specs.ratio23.is_fixed = True
    sextet_specs.magnetic_field.is_fixed = True
    sextet_specs.line_width1.is_fixed = True
    sextet_specs.line_width2.is_fixed = True
    sextet_specs.line_width3.is_fixed = True
    sextet_specs.line_width4.is_fixed = True
    sextet_specs.line_width5.is_fixed = True
    sextet_specs.line_width6.is_fixed = True
    sextet_orig = deepcopy(sextet_specs)
    flat_params = LSQModel._flatten_sextet_fit_params(sextet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    LSQModel._rebuild_sextet_from_fit_params(sextet_specs, iterator)
    assert sextet_specs == sextet_orig


def test_rebuild_sextet_from_fit_params_coupled():
    """Test if sextet is rebuilt correctly from flattened fit parameters - coupled case."""
    sextet_specs_coupled = deepcopy(SEXTET_SPECS)
    sextet_specs_coupled.line_width_coupling = LineWidthCoupling.COUPLED
    sextet_specs_coupled_orig = deepcopy(sextet_specs_coupled)
    coupled_flat = LSQModel._flatten_sextet_fit_params(sextet_specs_coupled)
    coupled_iterator = iter(list(map(lambda param: SpecsVar(param), coupled_flat)))
    LSQModel._rebuild_sextet_from_fit_params(sextet_specs_coupled, coupled_iterator)
    assert sextet_specs_coupled_orig == sextet_specs_coupled


def test_rebuild_sextet_from_fit_params_coupled_pairs():
    """Test if sextet is rebuilt correctly from flattened fit parameters - coupled pairs case."""
    sextet_specs_coupled_pairs = deepcopy(SEXTET_SPECS)
    sextet_specs_coupled_pairs.line_width_coupling = LineWidthCoupling.COUPLED_PAIRS
    sextet_specs_coupled_pairs_orig = deepcopy(sextet_specs_coupled_pairs)
    coupled_pairs_flat = LSQModel._flatten_sextet_fit_params(sextet_specs_coupled_pairs)
    coupled_pairs_iterator = iter(list(map(lambda param: SpecsVar(param), coupled_pairs_flat)))
    LSQModel._rebuild_sextet_from_fit_params(sextet_specs_coupled_pairs, coupled_pairs_iterator)
    assert sextet_specs_coupled_pairs_orig == sextet_specs_coupled_pairs


def test_rebuild_singlet_from_fit_params(lsq_model):
    """Test if singlet is rebuilt correctly from flattened fit parameters."""
    singlet_specs = deepcopy(SINGLET_SPECS)
    flat_params = lsq_model._flatten_fit_params(singlet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    lsq_model._rebuild_obj_from_fit_params(singlet_specs, iterator)
    assert singlet_specs == SINGLET_SPECS


def test_rebuild_singlet_from_fit_params_locked(lsq_model):
    """Test if singlet is rebuilt correctly from flattened fit parameters - locked case."""
    singlet_specs = deepcopy(SINGLET_SPECS)
    singlet_specs.amplitude.is_fixed = True  # .map changes the type, thus this awful locking
    singlet_specs.isomer_shift.is_fixed = True
    singlet_specs.line_width1.is_fixed = True
    singlet_orig = deepcopy(singlet_specs)
    flat_params = lsq_model._flatten_fit_params(singlet_specs)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    lsq_model._rebuild_obj_from_fit_params(singlet_specs, iterator)
    assert singlet_specs == singlet_orig


def test_rebuild_spectrum_from_fit_params(lsq_model):
    """Test if spectrum is rebuilt correctly from flattened fit parameters."""
    spectrum = deepcopy(ALL_SPECTRUM_SPECS)
    flat_params = lsq_model._flatten_fit_params(spectrum)
    iterator = iter(list(map(lambda param: SpecsVar(param), flat_params)))
    lsq_model._rebuild_obj_from_fit_params(spectrum, iterator)
    assert spectrum == ALL_SPECTRUM_SPECS


def test_rebuild_spectroscope_from_fit_params(lsq_model):
    """Test if spectroscope is rebuilt correctly from flattened fit parameters."""
    spectroscope = deepcopy(SPECTROSCOPE)
    flat_params = lsq_model._flatten_fit_params(spectroscope)
    iterator = iter(flat_params)
    lsq_model._rebuild_obj_from_fit_params(spectroscope, iterator)
    assert spectroscope == SPECTROSCOPE


def test_rebuild_from_fit_params(lsq_model):
    """Test if spectrum and spectroscope are rebuilt correctly from flattened fit parameters."""
    spectrum = deepcopy(ALL_SPECTRUM_SPECS)
    spectroscope = deepcopy(SPECTROSCOPE)
    flat_params = lsq_model._flatten_fit_params(spectrum) + lsq_model._flatten_fit_params(spectroscope)
    spectrum_rebuilt, spectroscope_rebuilt = lsq_model._rebuild_from_fit_params(spectrum, spectroscope, flat_params)
    assert spectrum_rebuilt == ALL_SPECTRUM_SPECS.map(lambda x: x.value)
    assert spectroscope_rebuilt == SPECTROSCOPE


# ------------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------------


def test_flatten_spectroscope_fit_params(lsq_calibration_model):
    """Test if spectroscope fit parameters are flattened correctly."""
    true = []
    true += [SPECTROSCOPE_SPECS.scale.value] if not SPECTROSCOPE_SPECS.scale.is_fixed else []
    true += [SPECTROSCOPE_SPECS.isomer_shift_ref.value] if not SPECTROSCOPE_SPECS.isomer_shift_ref.is_fixed else []
    flat_params = lsq_calibration_model._flatten_fit_params(SPECTROSCOPE_SPECS)
    assert flat_params == true


def test_rebuild_from_fit_params_calibration(lsq_calibration_model):
    """Test if spectrum and spectroscope are rebuilt correctly from flattened fit parameters."""
    spectrum = deepcopy(ALL_SPECTRUM_SPECS)
    spectroscope = deepcopy(SPECTROSCOPE_SPECS)
    flat_params = lsq_calibration_model._flatten_fit_params(spectrum) + lsq_calibration_model._flatten_fit_params(
        spectroscope
    )
    spectrum_rebuilt, spectroscope_rebuilt = lsq_calibration_model._rebuild_from_fit_params(
        spectrum, spectroscope, flat_params
    )
    assert spectrum_rebuilt == ALL_SPECTRUM_SPECS.map(lambda x: x.value)
    assert spectroscope_rebuilt == SPECTROSCOPE_SPECS.map(lambda x: x.value)
