"""Constants for PULS module."""

from enum import Enum


K = 10  # Number of experiments for each setting
RESULTS_DIR = "output"

MODELS = ["nnpu", "drpu"]
TC_METHODS = ["train", "true", "KM1", "KM2", "DR"]
PI_ESTIMATION_METHODS = ["KM1", "KM2", "DR"]
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "threshold",
    "estimated_test_pi",
]

class LabelShiftMethod(Enum, str):
    """Label shift handling methods.
    
    MLLS: Maximum Likelihood Label Shift
    TC: Threshold Correction
    """

    MLLS = "mlls"
    TC = "tc"