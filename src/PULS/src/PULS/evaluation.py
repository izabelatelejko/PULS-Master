"""Module for evaluation of PULS results."""

import json
from typing import List, Optional
import pandas as pd

from PULS.const import K, RESULTS_DIR, TC_METHODS, PI_ESTIMATION_METHODS, METRICS


def get_single_TC_metrics(
    dataset_name: str,
    mean: Optional[float],
    n: int,
    label_frequency: float,
    train_pi: float,
    test_pi: float,
    aggregate: bool = False,
    single_exp: bool = False,
):
    """Get combined TC metrics for the given dataset and parameters."""
    tc_results = {}
    print(TC_METHODS)
    for method in TC_METHODS:
        tc_results[method] = {}
        for metric in METRICS:
            tc_results[method][metric] = []

    n_exp = 1 if single_exp else K
    for exp_number in range(0, n_exp):
        metrics_file_path = f"{RESULTS_DIR}/{dataset_name}/{n}/"
        if mean is not None:
            metrics_file_path += f"{mean}/"
        metrics_file_path += (
            f"{train_pi}/{test_pi}/nnPUcc/{label_frequency}/{exp_number}/metrics.json"
        )
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
                tc_results[method][metric] = sum(tc_results[method][metric]) / n_exp

    return tc_results


def get_combined_TC_metrics(
    dataset_name: str,
    mean: Optional[float],
    n: int,
    label_frequency: float,
    train_pi: List[float],
    test_pi: List[float],
    aggregate: bool = False,
    single_exp: bool = False,
):
    """Get combined TC metrics for the given dataset and parameters."""

    combined_tc_metrics = {}

    for pi in train_pi:
        combined_tc_metrics[f"{pi}"] = {}
        for new_pi in test_pi:
            combined_tc_metrics[f"{pi}"][f"{new_pi}"] = get_single_TC_metrics(
                dataset_name,
                mean,
                n,
                label_frequency,
                pi,
                new_pi,
                aggregate=aggregate,
                single_exp=single_exp,
            )
    return combined_tc_metrics


def evaluate_single_shifted_pi_estimation(
    metrics: dict, true_shifted_pi: float, single_exp: bool
):
    """Evaluate TC metrics for given PULS setting."""

    pi_results = {}
    for method in PI_ESTIMATION_METHODS:
        pi_results[method] = {}

        if single_exp:
            pi_results[method]["estimated_test_pi"] = metrics[method][
                "estimated_test_pi"
            ][0]
        else:
            absolute_errors = [
                abs(pi - true_shifted_pi) for pi in metrics[method]["estimated_test_pi"]
            ]
            pi_results[method]["mae"] = sum(absolute_errors) / K
            pi_results[method]["std_mae"] = (
                sum((ae - pi_results[method]["mae"]) ** 2 for ae in absolute_errors)
                / (K - 1)
            ) ** 0.5
            pi_results[method]["mse"] = (
                sum(
                    (pi - true_shifted_pi) ** 2
                    for pi in metrics[method]["estimated_test_pi"]
                )
                / K
            )
            pi_results[method]["mean"] = sum(metrics[method]["estimated_test_pi"]) / K
            pi_results[method]["std"] = (
                sum(
                    (pi - pi_results[method]["mean"]) ** 2
                    for pi in metrics[method]["estimated_test_pi"]
                )
                / (K - 1)
            ) ** 0.5
            pi_results[method]["se"] = pi_results[method]["std"] / K

    return pi_results


def evaluate_shifted_pi_estimation(
    dataset_name: str,
    mean: Optional[float],
    n: int,
    label_frequency: float,
    train_pi: List[float],
    test_pi: List[float],
    convert_to_df: bool = False,
    single_exp: bool = False,
):
    """Evaluate TC metrics for given PULS setting."""

    combined_tc_metrics = get_combined_TC_metrics(
        dataset_name=dataset_name,
        mean=mean,
        n=n,
        label_frequency=label_frequency,
        train_pi=train_pi,
        test_pi=test_pi,
        single_exp=single_exp,
    )

    combined_pi_results = {}

    for pi in train_pi:
        combined_pi_results[f"{pi}"] = {}
        for new_pi in test_pi:
            combined_pi_results[f"{pi}"][f"{new_pi}"] = (
                evaluate_single_shifted_pi_estimation(
                    combined_tc_metrics[f"{pi}"][f"{new_pi}"], new_pi, single_exp
                )
            )

    if not convert_to_df:
        return combined_pi_results

    if single_exp:
        combined_pi_results_df = pd.DataFrame(
            columns=["pi", "new_pi", "method", "estimated_test_pi"]
        )
    else:
        combined_pi_results_df = pd.DataFrame(
            columns=[
                "pi",
                "new_pi",
                "method",
                "mae",
                "std_mae",
                "mse",
                "mean",
                "std",
                "se",
            ]
        )
    for pi in train_pi:
        for new_pi in test_pi:
            for method in PI_ESTIMATION_METHODS:
                if single_exp:
                    pi_results_row = {
                        "pi": pi,
                        "new_pi": new_pi,
                        "method": method,
                        "estimated_test_pi": combined_pi_results[f"{pi}"][f"{new_pi}"][
                            method
                        ]["estimated_test_pi"],
                    }
                else:
                    pi_results_row = {
                        "pi": pi,
                        "new_pi": new_pi,
                        "method": method,
                        "mae": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["mae"],
                        "std_mae": combined_pi_results[f"{pi}"][f"{new_pi}"][method][
                            "std_mae"
                        ],
                        "mse": combined_pi_results[f"{pi}"][f"{new_pi}"][method]["mse"],
                        "mean": combined_pi_results[f"{pi}"][f"{new_pi}"][method][
                            "mean"
                        ],
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
    train_pi: List[float],
    test_pi: List[float],
    convert_to_df: bool = False,
    single_exp: bool = False,
):
    """Evaluate TC metrics for all PULS settings."""

    combined_tc_metrics = get_combined_TC_metrics(
        dataset_name,
        mean,
        n,
        label_frequency,
        train_pi,
        test_pi,
        aggregate=True,
        single_exp=single_exp,
    )

    if not convert_to_df:
        return combined_tc_metrics

    combined_tc_metrics_df = pd.DataFrame(
        columns=["pi", "new_pi", "method", "metric", "average_value"]
    )
    for pi in train_pi:
        for new_pi in test_pi:
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
