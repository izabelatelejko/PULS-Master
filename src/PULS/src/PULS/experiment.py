"""Experiment class for PULS."""

import json
from typing import TYPE_CHECKING, Any, Optional
import numpy as np
import pkbar
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.optim import Adam

from DRPU.algorithm import priorestimator as ratio_estimator
from DRPU.algorithm import PUsequence, to_ndarray
from DRPU.modules.Kernel_MPE import KM1_KM2_estimate
from nnPU.dataset_configs import DatasetConfig
from nnPU.run_experiment import Experiment, DictJsonEncoder
from nnPU.experiment_config import ExperimentConfig
from nnPU.model import PUModel
from nnPU.loss import DRPUccLoss
# from PULS.const import LabelShiftMethod
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
        self.ratio_model = PUModel(self.n_inputs, activate_output=True)
        self.ratio_optimizer = Adam(
            self.ratio_model.parameters(),
            lr=self.experiment_config.dataset_config.learning_rate,
            weight_decay=0.005,
            betas=(0.9, 0.999),
        )
        self.ratio_train_metrics = []

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
        self.metrics["dataset_stats"] = {
            "train": data["train"].dataset_stats,
            "test": data["test"].dataset_stats,
        }

    def _train_step_ratio_estimator(self, epoch: int, kbar: pkbar.Kbar) -> None:
        """Train the ratio estimator model."""
        self.ratio_model.train()
        tr_loss = 0

        loss_fct = DRPUccLoss(prior=self.prior, alpha=None)
        for batch_idx, (data, _, label) in enumerate(self.train_loader):
            data, label = data.to(self.device), label.to(self.device)
            self.ratio_optimizer.zero_grad()
            output = self.ratio_model(data)

            loss = loss_fct(output.view(-1), label.type(torch.float))
            tr_loss += loss.item()
            loss.backward()
            self.ratio_optimizer.step()

            kbar.update(batch_idx + 1, values=[("loss", loss)])

        metric_values = {"tr_loss": tr_loss, "epoch": epoch}
        self.ratio_train_metrics.append(metric_values)

    def train_ratio_estimator(self) -> None:
        """Train the density ratio estimator model."""
        self._set_seed()
        self.ratio_model = self.ratio_model.to(self.device)

        for epoch in range(self.experiment_config.dataset_config.num_epochs):
            kbar = pkbar.Kbar(
                target=len(self.train_loader) + 1,
                epoch=epoch,
                num_epochs=self.experiment_config.dataset_config.num_epochs,
                width=8,
                always_stateful=False,
            )
            self._train_step_ratio_estimator(epoch, kbar)

        kbar = pkbar.Kbar(
            target=1,
            epoch=epoch,
            num_epochs=self.experiment_config.dataset_config.num_epochs,
            width=8,
            always_stateful=False,
        )

        with open(self.experiment_config.drpu_metrics_file, "w") as f:
            json.dump(self.ratio_train_metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.drpu_metrics_file)

    def _estimate_test_km_priors(self) -> tuple[float, float]:
        """Estimate the prior of test set with KM1 and KM2 methods."""
        pos = self.train_set.data.clone()[self.train_set.pu_targets == 1].numpy()
        unl = self.test_set.data.clone().numpy()
        KM1, KM2 = KM1_KM2_estimate(pos, unl)

        return KM1, KM2

    def _estimate_test_density_ratio_prior(self) -> float:
        """Estimate the prior of test set with density ratio method."""
        self.ratio_model.eval()
        with torch.no_grad():
            preds_P, preds_U = [], []

            # positive from training set
            for data, target, _ in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.ratio_model(data)  # .view(-1)
                preds = preds[target == 1]
                preds_P.append(to_ndarray(preds))  # to_ndarray(y))

            # unlabeled from shifted data
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                preds = self.ratio_model(data)  # .view(-1)
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

    
    def _run_mlls(
        self,
        model: torch.nn.Module,
        train_prior: float,
        tol: float = 1e-3,
        max_iter: int = 100,
        factor: Optional[float] = None,
    ) -> tuple[float, torch.Tensor]:
        """
        Estimate shifted test prior using Maximum Likelihood Label Shift (MLLS)
        with EM algorithm for the binary classification setting.
        """
        new_pi = train_prior
        old_pi = 0.0
        probs = torch.Tensor([])
        targets = torch.Tensor([])
        n_samples = 0

        model.eval()
        with torch.no_grad():
            for data, target, _ in self.test_loader:
                data = data.to(self.device)
                logits = model(data)
                if factor is not None:
                    logits = logits * factor
                batch_probs = torch.sigmoid(logits).view(-1).cpu()
                probs = torch.cat((probs, batch_probs), dim=0)
                targets = torch.cat((targets, target.view(-1).cpu()), dim=0)
                n_samples += data.size(0)
        
        for iteration in range(max_iter):
            # E-step: compute expected posteriors for all test samples
            # Apply MLLS formula for adjusted posteriors:
            # p'(y=1|x) = [ (π'/π) * p(y=1|x) ] /
            #             [ (π'/π)*p(y=1|x) + ((1-π')/(1-π))*(1-p(y=1|x)) ]
            adjusted_post_plus = (new_pi / train_prior) * probs
            adjusted_post_minus = ((1 - new_pi) / (1 - train_prior)) * (1 - probs)
            adjusted_post = adjusted_post_plus / (adjusted_post_plus + adjusted_post_minus)

            # denominator = nominator + ((1 - new_pi) / (1 - train_prior)) * (1 - probs)
            # adjusted_post = nominator / denominator

            print(f"Iteration {iteration+1}: π' = {new_pi}")
            print(adjusted_post)

            # M-step: update π' using mean of adjusted posteriors
            old_pi = new_pi
            new_pi = adjusted_post.mean().item()
            print(f"Updated π': {new_pi}")

            # Check convergence
            if abs(new_pi - old_pi) < tol:
                print(f"MLLS converged in {iteration+1} iterations.")
                break

        return new_pi, adjusted_post, targets, iteration + 1


    # def _scale_model(self, model: Any, factor: float) -> Any:
    #     """Scale the model output by a factor."""
        
    #     def scaled_model(x):
    #         return model(x) * factor
        
    #     return scaled_model


    def _test_with_threshold_correction(self, estimated_pi: Optional[float] = None, drpu: bool = False, calculate_roc_curve: bool = False):
        """Testing with Threshold Correction method."""
        self.model.eval()
        self.ratio_model.eval()

        if drpu:
            model = self.ratio_model
            factor = self.prior.item()
            # model = self._scale_model(model=self.ratio_model, factor=self.prior.item())
        else:
            model = self.model
            factor = 1

        # if label_shift_method == LabelShiftMethod.MLLS:
        #     model = self._get_mlls_model(posterior_estimator=model, prior=self.prior.item())
        # else:

        # assuming train PI is known
        if estimated_pi is not None:
            threshold = (self.prior.item() * (1 - estimated_pi)) / (
                (self.prior.item() + estimated_pi - 2 * self.prior.item() * estimated_pi)
            )
        else:
            # use default threshold 0.5 if no estimate is provided
            threshold = 0.5

        test_loss = 0
        correct = 0
        num_pos = 0

        test_points = []
        targets = []
        preds = []
        outputs1 = []
        outputs2 = []
        outputs3 = []
        y_scores = []

        kbar = pkbar.Kbar(
            target=len(self.test_loader) + 1,
            epoch=0,
            num_epochs=1,
            width=8,
            always_stateful=False,
        )

        with torch.no_grad():
            if drpu:
                test_loss_func = DRPUccLoss(prior=self.prior.item(), alpha=None)
            else:
                test_loss_func = self.experiment_config.PULoss(
                    prior=self.prior
                )  # priors not always known
            for data, target, _ in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data).view(-1)
                outputs1.append(output)
                # output = output * factor # multiplying by factor must be here

                # outputs.append(output)
                if drpu:
                    sigmoid_output = output * factor
                else:
                    sigmoid_output = torch.sigmoid(output) #* factor
                outputs3.append(sigmoid_output)
                if calculate_roc_curve:
                    y_scores.append(sigmoid_output)

                test_loss += test_loss_func(
                    output.view(-1), target.type(torch.float)
                ).item()

                pred = torch.where(
                    sigmoid_output < threshold,
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
        metric_values.threshold = threshold
        metric_values.true_test_pi = self.label_shift_config.test_prior

        if calculate_roc_curve:
            y_scores = torch.cat(y_scores).detach().cpu().numpy()
            print(f"{'drpu' if drpu else 'nnpu'} y_scores:")
            print(y_scores)
            fpr, tpr, thres = metrics.roc_curve(targets, y_scores, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            roc_curve = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": thres.tolist(),
                "roc_auc": roc_auc,
            }
            return metric_values, roc_curve
        
        # plot 3 densities
        import matplotlib.pyplot as plt
        outputs1 = torch.cat(outputs1).detach().cpu().numpy()
        # outputs2 = torch.cat(outputs2).detach().cpu().numpy()   
        outputs3 = torch.cat(outputs3).detach().cpu().numpy()
        plt.figure(figsize=(8,6))
        plt.hist(outputs1, bins=50, density=True, alpha=0.5, label='Raw outputs')
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'Output distributions with threshold ({"DRPU" if drpu else "NNPU"})')
        plt.show()

        # plt.figure(figsize=(8,6))
        # plt.hist(outputs2, bins=50, density=True, alpha=0.5, label='Scaled outputs')
        # plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        # plt.title(f'Output distributions with threshold ({"DRPU" if drpu else "NNPU"})')
        # plt.show()

        plt.figure(figsize=(8,6))
        plt.hist(outputs3, bins=50, density=True, alpha=0.5, label='Sigmoid outputs')
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.title(f'Output distributions with threshold ({"DRPU" if drpu else "NNPU"})')
        plt.show()

        return metric_values
    
    def _test_with_mlls(self, estimated_pi: float, probs: torch.Tensor, targets: list[torch.Tensor], drpu: bool = False):
        """Testing with MLLS method."""
        threshold = 0.5
        preds = torch.where(
            probs < threshold,
            torch.tensor(-1, device=self.device),
            torch.tensor(1, device=self.device),
        )
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy() 
        metric_values = self._calculate_metrics(targets, preds)
        metric_values.n = len(self.test_loader.dataset)
        metric_values.train_pi = self.prior.item()  
        metric_values.estimated_test_pi = estimated_pi
        metric_values.threshold = threshold
        metric_values.true_test_pi = self.label_shift_config.test_prior
        return metric_values


    def test_shifted(self) -> None:
        """Test the model on the shifted data."""
        self._estimate_test_pi()
        self.metrics["TC"] = {"nnpu": {}, "drpu": {}}
        self.metrics["MLLS"] = {"nnpu": {}, "drpu": {}}
        self.metrics["NOLS"] = {"nnpu": {}, "drpu": {}}

        self.test_pis.mlls_nnpu, mlls_nnpu_preds, targets, self.test_pis.n_iter_mlls_nnpu = self._run_mlls(model=self.model, train_prior=self.prior.item())
        self.test_pis.mlls_drpu, mlls_drpu_preds, _, self.test_pis.n_iter_mlls_drpu = self._run_mlls(model=self.ratio_model, train_prior=self.prior.item(), factor=self.prior.item())
        self.metrics["test_pis"] = self.test_pis.model_dump()

        self.metrics["TC"]["nnpu"]["train"], self.metrics["TC"]["nnpu"]["roc_curve"] = self._test_with_threshold_correction(
            self.prior.item(), calculate_roc_curve=True
        )
        if self.test_pis.true:
            self.metrics["TC"]["nnpu"]["true"] = self._test_with_threshold_correction(
                self.test_pis.true
            )
        self.metrics["TC"]["nnpu"]["KM1"] = self._test_with_threshold_correction(
            self.test_pis.km1
        )
        self.metrics["TC"]["nnpu"]["KM2"] = self._test_with_threshold_correction(
            self.test_pis.km2
        )
        self.metrics["TC"]["nnpu"]["DR"] = self._test_with_threshold_correction(
            self.test_pis.dre
        )
        self.metrics["TC"]["nnpu"]["mlls-nnpu"] = self._test_with_threshold_correction(
            self.test_pis.mlls_nnpu
        )
        self.metrics["TC"]["nnpu"]["mlls-drpu"] = self._test_with_threshold_correction(
            self.test_pis.mlls_drpu
        )

        self.metrics["TC"]["drpu"]["train"], self.metrics["TC"]["drpu"]["roc_curve"] = self._test_with_threshold_correction(
            self.prior.item(), drpu=True, calculate_roc_curve=True
        )
        if self.test_pis.true:
            self.metrics["TC"]["drpu"]["true"] = self._test_with_threshold_correction(
                self.test_pis.dre, drpu=True
            )
        self.metrics["TC"]["drpu"]["KM1"] = self._test_with_threshold_correction(
            self.test_pis.km1, drpu=True
        )
        self.metrics["TC"]["drpu"]["KM2"] = self._test_with_threshold_correction(
            self.test_pis.km2, drpu=True
        )
        self.metrics["TC"]["drpu"]["DR"] = self._test_with_threshold_correction(
            self.test_pis.dre, drpu=True
        )
        self.metrics["TC"]["drpu"]["mlls-nnpu"] = self._test_with_threshold_correction(
            self.test_pis.mlls_nnpu, drpu=True
        )
        self.metrics["TC"]["drpu"]["mlls-drpu"] = self._test_with_threshold_correction(
            self.test_pis.mlls_drpu, drpu=True
        )

        self.metrics["MLLS"]["nnpu"] = self._test_with_mlls(
            self.test_pis.mlls_nnpu, mlls_nnpu_preds, targets, drpu=False
        )
        self.metrics["MLLS"]["drpu"] = self._test_with_mlls(
            self.test_pis.mlls_drpu, mlls_drpu_preds, targets, drpu=True
        )

        self.metrics["NOLS"]["nnpu"] = self._test_with_threshold_correction()
        self.metrics["NOLS"]["drpu"] = self._test_with_threshold_correction(drpu=True)

        with open(self.experiment_config.metrics_file, "w") as f:
            json.dump(self.metrics, f, cls=DictJsonEncoder, indent=4)
        print("Metrics saved to", self.experiment_config.metrics_file)
