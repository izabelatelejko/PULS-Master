"""Module for dataset classes and functions."""

import numpy as np
from typing import Optional
import torch

from nnPU.dataset import (
    PUDatasetBase,
    BinaryTargetTransformer,
    PULabeler,
    MNIST_PU,
)

# class SyntheticGaussPUDataset(PUDatasetBase):

#     N = None
#     PI = None
#     MEAN = None

#     def __init__(
#         self,
#         root,
#         pu_labeler: PULabeler = None,
#         target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
#             included_classes=[1, -1], positive_classes=[1]
#         ),
#         train=True,
#         download=True,  # ignored
#         random_seed=None,
#     ) -> None:

#         assert self.N is not None
#         assert self.PI is not None
#         assert self.MEAN is not None

#         self.root = root
#         self.train = train
#         self.download = download
#         self.random_seed = random_seed
#         self.target_transformer = target_transformer
#         self.pu_labeler = pu_labeler

#         self.data = torch.cat(
#             [
#                 torch.normal(0, 1, (int(self.PI * self.N), 10)),
#                 torch.normal(self.MEAN, 1, (self.N - int(self.PI * self.N), 10)),
#             ]
#         )
#         self.targets = torch.cat(
#             [
#                 torch.ones(int(self.PI * self.N)),
#                 -1 * torch.ones(self.N - int(self.PI * self.N)),
#             ]
#         )

#         self._convert_to_pu_data()


# def generate_gauss_dataset_classes(
#     N: int, PI: float, PI_NEW: float, MEAN: float
# ) -> Dict[str, SyntheticGaussPUDataset]:
#     """
#     Dynamically creates a Synthetic Gauss PU dataset class with given parameters.

#     Args:
#     - N (int): The number of samples.
#     - PI (float): The prior probability of class 1 samples.
#     - PI_NEW (float): The new prior probability of class 1 samples.
#     - MEAN (float or array): The mean for the data distribution.

#     Returns:
#     - tuple: A tuple containing the two dynamically created classes (SyntheticPUDataset_train, SyntheticPUDataset_ls).
#     """
#     # Define class attributes dynamically
#     class_attrs_train = {"N": N, "PI": PI, "MEAN": MEAN}
#     class_attrs_ls = {"N": N, "PI": PI_NEW, "MEAN": MEAN}

#     # Dynamically create the new class
#     gauss_classes = dict()
#     gauss_classes["train"] = type(
#         "GaussPU_train", (SyntheticGaussPUDataset,), class_attrs_train
#     )
#     gauss_classes["ls"] = type("GaussPU_ls", (SyntheticGaussPUDataset,), class_attrs_ls)

#     return gauss_classes


class Gauss_PULS(PUDatasetBase):

    def __init__(
        self,
        root,
        pu_labeler: PULabeler = None,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=[1, -1], positive_classes=[1]
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.random_seed = random_seed
        self.target_transformer = target_transformer
        self.pu_labeler = pu_labeler

        self.data = torch.cat(
            [
                torch.normal(0, 1, (2000, 10)),
                torch.normal(0.5, 1, (2000, 10)),
            ]
        )
        self.targets = torch.cat(
            [
                torch.ones(2000),
                -1 * torch.ones(2000),
            ]
        )

        self._convert_to_pu_data()

        if shifted_prior is not None:
            self.shift_pu_data(shifted_prior, n_samples)


class MNIST_PULS(MNIST_PU):
    def __init__(
        self,
        root,
        pu_labeler: PULabeler,
        target_transformer: BinaryTargetTransformer = BinaryTargetTransformer(
            included_classes=np.arange(10),
            positive_classes=[1, 3, 5, 7, 9],
        ),
        train=True,
        download=True,  # ignored
        random_seed=None,
        shifted_prior: Optional[float] = None,
        n_samples: Optional[int] = None,
    ):
        super().__init__(
            root=root,
            pu_labeler=pu_labeler,
            target_transformer=target_transformer,
            train=train,
            download=download,
            random_seed=random_seed,
        )

        # If shifting parameters are provided, apply the shift
        if shifted_prior is not None:
            self.shift_pu_data(shifted_prior, n_samples)
