"""Constants for PULS module."""

K = 10  # Number of experiments for each setting
RESULTS_DIR = "output"

TC_METHODS = ["train", "true", "KM1", "KM2", "DRPU", "DRPU-true"]
PI_ESTIMATION_METHODS = ["KM1", "KM2", "DRPU"]
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "threshold",
    "estimated_test_pi",
]
