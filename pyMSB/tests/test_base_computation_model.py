from copy import deepcopy

import numpy as np
import pytest

from pyMSB.computation_models.base import AAnalysis, ACalibration, AModel
from pyMSB.exceptions import MossbauerExc, PhysicsValidationExc, ValidationExc
from pyMSB.models import LineWidthCoupling, SpectroscopeGeometry
from pyMSB.tests.conftest import (
    ANALYSIS_SPECS,
    CALIBRATION_SPECS,
)

# ------------------------------------------------------------------------------------
# Concrete class definition for testing purposes
# ------------------------------------------------------------------------------------


class ConcreteAModel(AModel):
    """Concrete class for testing AModel abstract class."""

    def __init__(self, *args, **kwargs):
        """Initialize the class."""
        super().__init__(*args, **kwargs)

    def _validate_specifications(self) -> None:
        pass

    def _fit(self) -> None:
        pass

    def _create_results(self) -> None:
        pass

    def predict(self, channels: np.ndarray | list[int]) -> None:
        pass


class ConcreteAAnalysis(AAnalysis):
    """Concrete class for testing AAnalysis abstract class."""

    def __init__(self, *args, **kwargs):
        """Initialize the class."""
        super().__init__(*args, **kwargs)

    def _fit(self) -> None:
        pass

    def _create_results(self) -> None:
        pass

    def predict(self, channels: np.ndarray | list[int]) -> None:
        pass


class ConcreteACalibration(ACalibration):
    """Concrete class for testing ACalibration abstract class."""

    def __init__(self, *args, **kwargs):
        """Initialize the class."""
        super().__init__(*args, **kwargs)

    def _fit(self) -> None:
        pass

    def _create_results(self) -> None:
        pass

    def predict(self, channels: np.ndarray | list[int]) -> None:
        pass


# ------------------------------------------------------------------------------------
# AModel
# ------------------------------------------------------------------------------------


def test_validate_data():
    """Test _validate_data method."""
    a_model = ConcreteAModel(specs=None, geometry=SpectroscopeGeometry.REFLECTION, strict=False)
    a_model._validate_data(channels=[0, 1, 2], counts=[0, 1, 0])


def test_validate_data_invalid_length():
    """Test _validate_data method with invalid data lengths."""
    with pytest.raises(ValidationExc):
        a_model = ConcreteAModel(specs=None, geometry=SpectroscopeGeometry.REFLECTION, strict=False)
        a_model._validate_data(channels=[0, 1, 2, 3], counts=[0, 1, 0])


def test_validate_data_invalid_geometry():
    """Test _validate_data method with invalid geometry."""
    with pytest.raises(ValidationExc):
        a_model = ConcreteAModel(specs=None, geometry=SpectroscopeGeometry.TRANSMISSION, strict=False)
        a_model._validate_data(channels=[0, 1, 2], counts=[0, 1, 0])


def test_results_invalid():
    """Test results property with invalid results."""
    with pytest.raises(MossbauerExc):
        a_model = ConcreteAModel(specs=None, geometry=SpectroscopeGeometry.TRANSMISSION, strict=False)
        _ = a_model.results


# ------------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------------


def test_validate_specifications():
    """Test _validate_specifications method."""
    ConcreteAAnalysis(specs=deepcopy(ANALYSIS_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


def test_validate_specifications_invalid():
    """Test _validate_specifications method with invalid specs."""
    with pytest.raises(ValidationExc):
        ConcreteAAnalysis(specs=deepcopy(CALIBRATION_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


def test_analyze_physics():
    """Test analyze_physics method."""
    problems = AAnalysis.analyze_physics(specs=deepcopy(ANALYSIS_SPECS))
    assert len(problems) == 0


def test_analyze_physics_invalid_line_coupling_logic():
    """Test analyze_physics method with invalid line coupling logic."""
    invalid_analysis_specs = deepcopy(ANALYSIS_SPECS)
    invalid_analysis_specs.spectrum.doublets[0].line_width_coupling = LineWidthCoupling.COUPLED
    invalid_analysis_specs.spectrum.doublets[0].line_width1.is_fixed = True
    invalid_analysis_specs.spectrum.doublets[0].line_width2.is_fixed = False
    problems = AAnalysis.analyze_physics(specs=invalid_analysis_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid():
    """Test _analyze_physics method with invalid specs (UNLIKE THE TEST ABOVE, TESTS IF EXCEPTION IS RISEN)."""
    with pytest.raises(PhysicsValidationExc):
        invalid_analysis_specs = deepcopy(ANALYSIS_SPECS)
        invalid_analysis_specs.spectrum.doublets[0].line_width_coupling = LineWidthCoupling.COUPLED
        invalid_analysis_specs.spectrum.doublets[0].line_width1.is_fixed = True
        invalid_analysis_specs.spectrum.doublets[0].line_width2.is_fixed = False
        ConcreteAAnalysis(specs=invalid_analysis_specs, geometry=SpectroscopeGeometry.REFLECTION)


# ------------------------------------------------------------------------------------
# Calibration
# ------------------------------------------------------------------------------------


def test_validate_specifications_calibration():
    """Test _validate_specifications method for calibration."""
    ConcreteACalibration(specs=deepcopy(CALIBRATION_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


def test_validate_specifications_invalid_calibration():
    """Test _validate_specifications method with invalid specs for calibration."""
    with pytest.raises(ValidationExc):
        ConcreteACalibration(specs=deepcopy(ANALYSIS_SPECS), geometry=SpectroscopeGeometry.REFLECTION)


def test_analyze_physics_calibration():
    """Test analyze_physics method for calibration."""
    problems = ACalibration.analyze_physics(specs=deepcopy(CALIBRATION_SPECS))
    assert len(problems) == 0


def test_analyze_physics_invalid_sextet_calibration():
    """Test analyze_physics method with invalid sextet specs for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.spectrum.sextets = []
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 3


def test_analyze_physics_invalid_magnetic_field_calibration():
    """Test analyze_physics method with invalid magnetic field specs for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.spectrum.sextets[0].magnetic_field.is_fixed = False
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid_line_coupling_logic_calibration():
    """Test analyze_physics method with invalid line coupling logic for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.spectrum.sextets[0].line_width_coupling = LineWidthCoupling.COUPLED
    invalid_calibration_specs.spectrum.sextets[0].line_width1.is_fixed = True
    invalid_calibration_specs.spectrum.sextets[0].line_width2.is_fixed = False
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid_isomer_shift_calibration():
    """Test analyze_physics method with invalid isomer shift specs for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.spectrum.sextets[0].isomer_shift.is_fixed = False
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid_scale_calibration():
    """Test analyze_physics method with invalid scale specs for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.scope.scale.is_fixed = True
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid_isomer_shift_ref_calibration():
    """Test analyze_physics method with invalid isomer shift ref specs for calibration."""
    invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
    invalid_calibration_specs.scope.isomer_shift_ref.is_fixed = True
    problems = ACalibration.analyze_physics(specs=invalid_calibration_specs)
    assert len(problems) == 1


def test_analyze_physics_invalid_calibration():
    """Test _analyze_physics method with invalid specs for calibration."""
    with pytest.raises(PhysicsValidationExc):
        invalid_calibration_specs = deepcopy(CALIBRATION_SPECS)
        invalid_calibration_specs.scope.isomer_shift_ref.is_fixed = True
        ConcreteACalibration(specs=invalid_calibration_specs, geometry=SpectroscopeGeometry.REFLECTION)
