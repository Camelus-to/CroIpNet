from datetime import datetime
import torch.utils.data as utils
import numpy as np
# import wandb
import hydra
from omegaconf import DictConfig, open_dict
from dataset import dataset_factory
from models import model_factory, test_model_factory
from components import lr_scheduler_factory, optimizers_factory, logger_factory
from training import training_factory
from datetime import datetime
import pandas as pd
import torch
from test_model import Tester
import pickle
device = 'cuda'


def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def model_training(cfg: DictConfig):
    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")
    # Select different language encodings for different diseases
    cfg.dataset.name = 'abideII' # Available models and datasets: abideII, abide, adhd
    if cfg.dataset.name == 'abide':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)
        test_dataset_path = r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-lstm_tranformer\dataset\Test_dataset\test_dataset_abide.pkl'

    elif cfg.dataset.name == 'adhd':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)
        test_dataset_path = r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-lstm_tranformer\dataset\Test_dataset\test_dataset_adhd.pkl'
    elif cfg.dataset.name == 'abideII':
        semantic_similarity_matrix = pd.read_csv(
            'E:\Technolgy_learning\Learning_code\AD\AD_Bert\获取文本\similarity_matrix.csv', header=None)
        test_dataset_path = r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-lstm_tranformer\dataset\Test_dataset\test_dataset_abideII.pkl'
    # Convert Pandas DataFrame to NumPy array
    semantic_similarity_numpy = semantic_similarity_matrix.values  # DataFrame to NumPy array

    # Convert NumPy array to PyTorch Tensor
    semantic_similarity_matrix = torch.tensor(semantic_similarity_numpy, dtype=torch.float32)
    semantic_similarity_matrix = semantic_similarity_matrix.to(device)
    # The three parameters node_sz, node_feature_sz, and timeseries_sz are obtained through dataset_factory
    dataloaders, node_size, node_feature_size, timeseries_size, datasets = dataset_factory(cfg)
    logger = logger_factory(cfg)

    test_dataset = load_data(test_dataset_path)

    # Directly load the test set data here
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=8, shuffle=True, drop_last=False)
    model = test_model_factory(cfg, semantic_similarity_matrix, cfg.dataset.name)
    path = fr"E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-lstm_tranformer\models\Test\best_model_{cfg.dataset.name}.pt",  # Model parameter file path
    tester = Tester(model=model,
                    test_dataloader=test_dataloader,
                    path=path,  # Model parameter file path
                    config=cfg,
                    semantic_similarity_matrix=semantic_similarity_matrix)
    results = tester.run_test()

    optimizers = optimizers_factory(
        model=model, optimizer_configs=cfg.optimizer)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger, semantic_similarity_matrix)

    return training.train()


# @hydra.main(version_base=None, config_path="conf", config_name="config")
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"
    # _{cfg.training.name}\
    # _{cfg.optimizer[0].lr_scheduler.mode}"
    # Five-fold cross-validation is performed here
    all_fold_results = []
    for _ in range(cfg.repeat_time):
        cfg.lunshu = _
        best_result = model_training(cfg)
        all_fold_results.append(best_result)
    # Calculate the average value of each metric
    metrics = list(all_fold_results[0].keys())
    avg_results = {metric: np.mean([fold_result[metric] for fold_result in all_fold_results]) for metric in metrics}

    # Print the final average results
    print("\nFinal Average Results Across All Folds:")
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}")


if __name__ == '__main__':
    main()
