"""Module for evaluation of PULS results."""

import json
from typing import List
import pandas as pd

from PULS.const import K, RESULTS_DIR, TC_METHODS, PI_ESTIMATION_METHODS, METRICS


def get_single_TC_metrics(
    dataset_name: str,
    mean: float,
    n: int,
    label_frequency: float,
    pi: float,
    new_pi: float,
    aggregate: bool = False,
):
    """Get combined TC metrics for the given dataset and parameters."""
    tc_results = {}
    for method in TC_METHODS:
        tc_results[method] = {}
        for metric in METRICS:
            tc_results[method][metric] = []

    for exp_number in range(0, K):
        metrics_file_path = f"{RESULTS_DIR}/{dataset_name}/{n}/{mean}/{pi}/{new_pi}/nnPUcc/{label_frequency}/{exp_number}/metrics.json"
        with open(metrics_file_path, "r") as f:
            metrics_contents = json.load(f)

        for method in TC_METHODS:
            for metric in METRICS:
                tc_results[method][metric].append(
                    metrics_contents["TC"][method][metric]
                )

    if aggregate:
        for method in TC_METHODS:
            for metric in METRICS:
                tc_results[method][metric] = sum(tc_results[method][metric]) / K

    return tc_results


def get_combined_TC_metrics(
    dataset_name: str,
    mean: float,
    n: int,
    label_frequency: float,
    pi_grid: List[float],
    aggregate: bool = False,
):
    """Get combined TC metrics for the given dataset and parameters."""

    combined_tc_metrics = {}

    for pi in pi_grid:
        combined_tc_metrics[f"{pi}"] = {}
        for new_pi in pi_grid:
            combined_tc_metrics[f"{pi}"][f"{new_pi}"] = get_single_TC_metrics(
                dataset_name,
                mean,
                n,
                label_frequency,
                pi,
                new_pi,
                aggregate=aggregate,
            )
    return combined_tc_metrics


def evaluate_single_shifted_pi_estimation(metrics: dict, true_shifted_pi: float):
    """Evaluate TC metrics for given PULS setting."""

    pi_results = {}
    for method in PI_ESTIMATION_METHODS:
        pi_results[method] = {}

        absolute_errors = [
            abs(pi - true_shifted_pi) for pi in metrics[method]["estimated_shifted_pi"]
        ]
        pi_results[method]["mae"] = sum(absolute_errors) / K
        pi_results[method]["std_mae"] = (
            sum((ae - pi_results[method]["mae"]) ** 2 for ae in absolute_errors)
            / (K - 1)
        ) ** 0.5
        pi_results[method]["mse"] = (
            sum(
                (pi - true_shifted_pi) ** 2
                for pi in metrics[method]["estimated_shifted_pi"]
            )
            / K
        )
        pi_results[method]["mean"] = sum(metrics[method]["estimated_shifted_pi"]) / K
        pi_results[method]["std"] = (
            sum(
                (pi - pi_results[method]["mean"]) ** 2
                for pi in metrics[method]["estimated_shifted_pi"]
            )
            / (K - 1)
        ) ** 0.5
        pi_results[method]["se"] = pi_results[method]["std"] / K

    return pi_results


def evaluate_shifted_pi_estimation(
    dataset_name: str,
    mean: float,
    n: int,
    label_frequency: float,
    pi_grid: List[float],
    convert_to_df: bool = False,
):
    """Evaluate TC metrics for given PULS setting."""

    combined_tc_metrics = get_combined_TC_metrics(
        dataset_name,
        mean,
        n,
        label_frequency,
        pi_grid,
    )

    combined_pi_results = {}

    for pi in pi_grid:
        combined_pi_results[f"{pi}"] = {}
        for new_pi in pi_grid:
            combined_pi_results[f"{pi}"][f"{new_pi}"] = (
                evaluate_single_shifted_pi_estimation(
                    combined_tc_metrics[f"{pi}"][f"{new_pi}"], new_pi
                )
            )

    if not convert_to_df:
        return combined_pi_results

    combined_pi_results_df = pd.DataFrame(
        columns=["pi", "new_pi", "method", "mae", "std_mae", "mse", "mean", "std", "se"]
    )
    for pi in pi_grid:
        for new_pi in pi_grid:
            for method in ["KM1", "KM2", "DRE"]:
                pi_results_row = {
                    "pi": pi,
                    "new_pi": new_pi,
                    "method": method,
                    "mae": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["mae"],
                    "std_mae": combined_pi_results[f"{pi}"][f"{new_pi}"][method][
                        "std_mae"
                    ],
                    "mse": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["mse"],
                    "mean": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["mean"],
                    "std": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["std"],
                    "se": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["se"],
                }
                combined_pi_results_df = pd.concat(
                    [
                        combined_pi_results_df,
                        pd.DataFrame(pi_results_row, index=[0]),
                    ],
                    ignore_index=True,
                )

    return combined_pi_results_df


def evaluate_all_TC_metrics(
    dataset_name: str,
    mean: float,
    n: int,
    label_frequency: float,
    pi_grid: List[float],
    convert_to_df: bool = False,
):
    """Evaluate TC metrics for all PULS settings."""

    combined_tc_metrics = get_combined_TC_metrics(
        dataset_name,
        mean,
        n,
        label_frequency,
        pi_grid,
        aggregate=True,
    )

    if not convert_to_df:
        return combined_tc_metrics

    combined_tc_metrics_df = pd.DataFrame(
        columns=["pi", "new_pi", "method", "metric", "average_value"]
    )
    for pi in pi_grid:
        for new_pi in pi_grid:
            for method in TC_METHODS:
                for metric in METRICS:
                    tc_metrics_row = {
                        "pi": pi,
                        "new_pi": new_pi,
                        "method": method,
                        "metric": metric,
                        "average_value": combined_tc_metrics[f"{pi}"][f"{new_pi}"][
                            method
                        ][metric],
                    }
                    combined_tc_metrics_df = pd.concat(
                        [
                            combined_tc_metrics_df,
                            pd.DataFrame(tc_metrics_row, index=[0]),
                        ],
                        ignore_index=True,
                    )

        return combined_tc_metrics_df
