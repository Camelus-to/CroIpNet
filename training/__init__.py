from operator import mod
from .training import Train
from .FBNettraining import FBNetTrain
from omegaconf import DictConfig
from .prepossess import Utils
from typing import List
import torch
from source.components import LRScheduler
import logging
import torch.utils.data as utils


def training_factory(config: DictConfig,
                     model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger,
                     semantic_similarity_matrix: torch.Tensor) -> Train:
    train = config.model.get("train", None)  # train经过这一步是Train
    if not train:
        train = config.training.name
    return eval(train)(cfg=config,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger,
                       semantic_similarity_matrix=semantic_similarity_matrix)
