"""Experiment class for PULS."""

import json
from typing import TYPE_CHECKING
import numpy as np
import pkbar
import torch
from torch.utils.data import DataLoader

from DRPU.algorithm import priorestimator as ratio_estimator
from DRPU.algorithm import PUsequence, to_ndarray
from DRPU.modules.Kernel_MPE import KM1_KM2_estimate
from nnPU.dataset_configs import DatasetConfig
from nnPU.run_experiment import Experiment, DictJsonEncoder
from nnPU.experiment_config import ExperimentConfig
from PULS.models import PiEstimates

if TYPE_CHECKING:
    from PULS.models import LabelShiftConfig


class PULSExperiment(Experiment):
    """Experiment class for PULS data."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        label_shift_config: "LabelShiftConfig",
    ) -> None:
        """Initialize the PULS experiment."""
        self.metrics = {}
        self.label_shift_config = label_shift_config
        super().__init__(experiment_config)

        # self.enhanced_experiment_config = enhanced_experiment_config
        # self._prepare_shifted_data()

    def _prepare_data(self):
        self._set_seed()

        data = {}
        data["train"] = self.experiment_config.dataset_config.DatasetClass(
            self.experiment_config.data_dir,
            self.experiment_config.dataset_config.PULabelerClass(
                label_frequency=self.experiment_config.label_frequency
            ),
            train=True,
            download=True,
            random_seed=self.experiment_config.seed,
            shifted_prior=self.label_shift_config.train_prior,
            n_samples=self.label_shift_config.train_n_samples,
        )
        data["test"] = self.experiment_config.dataset_config.DatasetClass(
            self.experiment_config.data_dir,
            self.experiment_config.dataset_config.PULabelerClass(label_frequency=0),
            train=False,
            download=True,
            random_seed=self.experiment_config.seed,
            shifted_prior=self.label_shift_config.test_prior,
            n_samples=self.label_shift_config.test_n_samples,
        )

        self.train_set = data["train"]
        self.prior = self.train_set.get_prior()
        print("Train set prior:", self.prior.item())
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.experiment_config.dataset_config.train_batch_size,
            shuffle=True,
        )
        self.n_inputs = len(next(iter(self.train_set))[0].reshape(-1))

        self.test_set = test_set = data["test"]
        self.test_loader = DataLoader(
            test_set,
            batch_size=self.experiment_config.dataset_config.eval_batch_size,
            shuffle=False,
        )

    def _estimate_test_km_priors(self) -> tuple[float, float]:
        """Estimate the prior of test set with KM1 and KM2 methods."""
        pos = self.train_set.data.clone()[self.train_set.pu_targets == 1].numpy()
        unl = self.test_set.data.clone().numpy()
        KM1, KM2 = KM1_KM2_estimate(pos, unl)

        return KM1, KM2

    def _estimate_test_density_ratio_prior(self) -> float:
        """Estimate the prior of test set with density ratio method."""
        self.model.eval()
        with torch.no_grad():
            preds_P, preds_U = [], []

            # positive from training set
            for data, target, _ in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.model(data)  # .view(-1)
                preds = preds[target == 1]
                preds_P.append(to_ndarray(preds))  # to_ndarray(y))

            # unlabeled from shifted data
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.model(data)  # .view(-1)
                preds_U.append(to_ndarray(preds))  # to_ndarray(y))

            preds_P = np.concatenate(preds_P)
            preds_U = np.concatenate(preds_U)

            prior = ratio_estimator(
                np.concatenate([preds_P, preds_U]),
                PUsequence(len(preds_P), len(preds_U)),
            )

        return prior

    def _estimate_test_pi(self) -> None:
        """Estimate the test pi values using KM1, KM2, and DRE methods."""
        # True pi
        true_pi = self.label_shift_config.test_prior

        # KM1, KM2
        KM1, KM2 = self._estimate_test_km_priors()

        # Density ratio
        ratio_pi = self._estimate_test_density_ratio_prior()

        self.test_pis = PiEstimates(
            true=true_pi,
            km1=KM1,
            km2=KM2,
            dre=ratio_pi,
        )

    def _test_TC(self, estimated_pi: float):
        """Testing with Threshold Correction method."""
        self.model.eval()
        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []
        outputs = []

        kbar = pkbar.Kbar(
            target=len(self.test_loader) + 1,
            epoch=0,
            num_epochs=1,
            width=8,
            always_stateful=False,
        )

        # assuming train PI is known
        treshold = (self.prior.item() / (1 - self.prior.item())) * (
            (1 - estimated_pi) / estimated_pi
        )

        with torch.no_grad():
            test_loss_func = self.experiment_config.PULoss(
                prior=self.prior
            )  # priors not always known
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                outputs.append(output)
                sigmoid_output = torch.sigmoid(output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()

                pred = torch.where(
                    sigmoid_output / (1 - sigmoid_output) < treshold,
                    torch.tensor(-1, device=self.device),
                    torch.tensor(1, device=self.device),
                )
                num_pos += torch.sum(pred == 1)
                correct += pred.eq(target.view_as(pred)).sum().item()

                test_points.append(data)
                targets.append(target)
                preds.append(pred)

        test_loss /= len(self.test_loader)
        pos_fraction = float(num_pos) / len(self.test_loader.dataset)

        kbar.add(
            1,
            values=[
                ("test_loss", test_loss),
                ("accuracy", 100.0 * correct / len(self.test_loader.dataset)),
                ("pos_fraction", pos_fraction),
            ],
        )

        targets = torch.cat(targets).detach().cpu().numpy()
        preds = torch.cat(preds).detach().cpu().numpy()

        metric_values = self._calculate_metrics(targets, preds)
        metric_values.n = len(self.test_loader.dataset)
        metric_values.train_pi = self.prior.item()
        metric_values.estimated_test_pi = estimated_pi
        metric_values.treshold = treshold
        metric_values.true_test_pi = self.label_shift_config.test_prior

        return metric_values

    def test_shifted(self) -> None:
        """Test the model on the shifted data."""
        self._estimate_test_pi()
        self.metrics["TC"] = {}

        self.metrics["TC"]["train"] = self._test_TC(self.prior.item())
        if self.test_pis.true:
            self.metrics["TC"]["true"] = self._test_TC(self.test_pis.true)
        self.metrics["TC"]["KM1"] = self._test_TC(self.test_pis.km1)
        self.metrics["TC"]["KM2"] = self._test_TC(self.test_pis.km2)
        self.metrics["TC"]["DRE"] = self._test_TC(self.test_pis.dre)

        with open(self.experiment_config.metrics_file, "w") as f:
            json.dump(self.metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.metrics_file)
