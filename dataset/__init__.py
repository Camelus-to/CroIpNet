from omegaconf import DictConfig, open_dict

from .abide import load_abide_data
from .adni import load_adni_data
from .dataloader import  init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    dataloaders = init_stratified_dataloader(cfg)\


    return dataloaders
