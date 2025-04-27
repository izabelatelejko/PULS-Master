"""Constants for PULS module."""

K = 10  # Number of experiments for each setting
RESULTS_DIR = "output"

TC_METHODS = ["train", "true", "KM1", "KM2", "DRE"]
PI_ESTIMATION_METHODS = ["KM1", "KM2", "DRE"]
METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "treshold",
    "estimated_test_pi",
]
