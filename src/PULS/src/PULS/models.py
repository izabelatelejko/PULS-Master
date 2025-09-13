"""Module for the PULS experiment configuration models."""

from typing import Optional
from pydantic import BaseModel


class PiEstimates(BaseModel):

    true: Optional[float] = None
    km1: Optional[float] = None
    km2: Optional[float] = None
    dre: Optional[float] = None
    mlls_nnpu: Optional[float] = None
    mlls_drpu: Optional[float] = None
    n_iter_mlls_nnpu: Optional[int] = None
    n_iter_mlls_drpu: Optional[int] = None


class LabelShiftConfig(BaseModel):

    train_prior: Optional[float] = None
    train_n_samples: Optional[int] = None
    test_prior: Optional[float] = None
    test_n_samples: Optional[int] = None


# class GaussExperimentConfig(BaseModel):

#     N: int
#     PI: float
#     PI_NEW: float
#     MEAN: float
#     label_frequency: float
#     exp_number: int

#     @cached_property
#     def dataset_class(self) -> Dict[str, SyntheticGaussPUDataset]:
#         """Return the dataset classes for the experiment."""
#         return generate_gauss_dataset_classes(
#             N=self.N, PI=self.PI, PI_NEW=self.PI_NEW, MEAN=self.MEAN
#         )

#     @cached_property
#     def data(self) -> Dict[str, SyntheticGaussPUDataset]:
#         """Return the dataset for the experiment."""
#         data = dict()
#         data["train"] = self.dataset_class["train"](
#             root="data",
#             train=True,
#             pu_labeler=SCAR_CC_Labeler(label_frequency=self.label_frequency),
#         )
#         data["ls"] = self.dataset_class["ls"](
#             root="data",
#             train=False,
#             pu_labeler=SCAR_CC_Labeler(label_frequency=self.label_frequency),
#         )
#         return data

#     @cached_property
#     def dataset_config(self) -> Dict[str, DatasetConfig]:
#         """Return the dataset configuration."""
#         dataset_configs = dict()
#         dataset_configs["train"] = DatasetConfig(
#             f"Gauss/{self.N}/{self.MEAN}/{self.PI}/{self.PI_NEW}",
#             DatasetClass=self.dataset_class["train"],
#             PULabelerClass=SCAR_CC_Labeler,
#         )
#         dataset_configs["ls"] = DatasetConfig(
#             f"Gauss/{self.N}/{self.MEAN}/{self.PI_NEW}/{self.PI_NEW}",
#             DatasetClass=self.dataset_class["ls"],
#             PULabelerClass=SCAR_CC_Labeler,
#         )
#         return dataset_configs

#     @cached_property
#     def experiment_config(self) -> ExperimentConfig:
#         """Return the experiment configuration."""
#         return ExperimentConfig(
#             PULoss=nnPUccLoss,
#             dataset_config=self.dataset_config["train"],
#             label_frequency=self.label_frequency,
#             exp_number=self.exp_number,
#         )
