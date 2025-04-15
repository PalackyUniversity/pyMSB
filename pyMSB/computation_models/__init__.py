import sys

from .base import AAnalysis, ACalibration
from .bayesian_model import BayesianCalibrationModel, BayesianModel, MAPCalibrationModel, MAPModel
from .lsq_model import LSQCalibrationModel, LSQModel

calibration_classes = []
analysis_classes = []
current_module = sys.modules[__name__]
for key in dir(current_module):
    cls = getattr(current_module, key)
    if not isinstance(cls, type):
        continue  # Skip non-class objects
    if issubclass(cls, ACalibration) and cls is not ACalibration:
        calibration_classes.append(cls)
    elif issubclass(cls, AAnalysis) and cls is not AAnalysis:
        analysis_classes.append(cls)
calibration_models = {cls.name: cls for cls in calibration_classes}
analysis_models = {cls.name: cls for cls in analysis_classes}


__all__ = [
    "BayesianModel",
    "BayesianCalibrationModel",
    "MAPModel",
    "MAPCalibrationModel",
    "LSQModel",
    "LSQCalibrationModel",
    "calibration_models",
    "analysis_models",
]
