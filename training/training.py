from source.utils import accuracy, TotalMeter, count_params, isfloat

import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data
# import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
from sklearn.metrics import confusion_matrix

device = 'cuda'


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_score = None
        self.epochs_no_improve = 0
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss  # 这里以验证集损失作为判断依据

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.epochs_no_improve += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.epochs_no_improve}/{self.patience}')
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.epochs_no_improve = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)


class Train:
    # 会执行init方法，
    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger,
                 semantic_similarity_matrix: torch.Tensor) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(f'#model params: {count_params(self.model)}')
        self.semantic_similarity_matrix = semantic_similarity_matrix
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, \
            self.test_loss, self.train_accuracy, \
            self.val_accuracy, self.test_accuracy = [
            TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        semantic_similarity_matrix = self.semantic_similarity_matrix
        for final_pearson, signals, label, pseudo, ages, genders, site in self.train_dataloader:
            label = label.float()  # label还是one-hot编码呢
            final_pearson, signals, label, pseudo, ages, genders, site = final_pearson.to(device), signals.to(
                device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device), site.to(device)
            self.current_step += 1

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            [output, score], data_graph, edge_variance, contrastive_loss = self.model(final_pearson, signals, pseudo,
                                                                                      ages, genders, site, label)
            alpha = 0.6
            # label = label.long()
            loss = self.loss_fn(output, label)
            loss = loss + contrastive_loss
            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            # loss.backward()
            optimizer.step()
            top1 = accuracy(output, label[:, 1])[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()
        semantic_similarity_matrix = self.semantic_similarity_matrix
        for final_pearson, signals, label, pseudo, ages, genders, site in dataloader:
            # time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            final_pearson, signals, label, pseudo, ages, genders, site = final_pearson.to(device), signals.to(
                device), label.to(device), pseudo.to(device), ages.to(device), genders.to(device), site.to(device)
            [output, score], data_graph, edge_variance, contrastive_loss = self.model(final_pearson, signals, pseudo,
                                                                                      ages, genders, site, label)
            # label = label.float()
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}")
            loss = self.loss_fn(output, label)
            loss = loss + contrastive_loss
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label[:, 1])[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label[:, 1].tolist()

        auc = roc_auc_score(labels, result)


        result, labels = np.array(result), np.array(labels)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0

        metric = precision_recall_fscore_support(
            labels, result, average='micro')

        return [auc] + list(metric)

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            # time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            time_series, node_feature, label = time_series, node_feature, label
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path / "training_process.npy",
                results, allow_pickle=True)

        torch.save(self.model.state_dict(), self.save_path / "model.pt")

    def train(self):
        training_process = []
        early_stopping = EarlyStopping(patience=2, verbose=True, path=self.save_path / "best_model.pt")  # 设定早停策略
        self.current_step = 0
        best_test_accuracy = 0.0
        best_test_result = {}
        for epoch in range(self.epochs):  # epochs是200轮
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])  # 单独训练每一轮
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            if self.test_accuracy.avg > best_test_accuracy:
                best_test_accuracy = self.test_accuracy.avg
                best_test_result = {
                    'Epoch': epoch,
                    'Best Test Accuracy': best_test_accuracy,
                    'Best Test AUC': test_result[0],
                    'Best Test Precision': test_result[1],
                    'Best Test Recall': test_result[2],
                    'Best Test F1-score': test_result[3],
                    'Best Test Sensitivity': test_result[-2],
                    'Best Test Specificity': test_result[-1]
                }
            # 只有在这个里面会打印一下，其余的都是会记录下来
            self.logger.info(" | ".join([
                f'Lunshu[{self.config.lunshu}]',
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.4f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test Precision:{test_result[1]:.4f}',
                f'Test Recall:{test_result[2]:.4f}',
                f'Test F1-score:{test_result[3]:.4f}',
                f'Test Spe:{test_result[-1]:.4f}',
                f'Test Sen:{test_result[-2]:.4f}',
                f'LR:{self.lr_schedulers[0].lr:.4f}'
            ]))

            training_process.append({
                "Epoch": epoch,
                "Train Loss": self.train_loss.avg,
                "Val Loss": self.val_loss.avg,
                "Val AUC": val_result[0],
                "Train Accuracy": self.train_accuracy.avg,
                "Test Loss": self.test_loss.avg,
                "Test Accuracy": self.test_accuracy.avg,
                "Test AUC": test_result[0],
                'micro F1': test_result[3],
                'Test Sensitivity': test_result[-2],
                'Test Specificity': test_result[-1],
            })
            # 检查是否满足早停条件
            early_stopping(self.val_loss.avg, self.model)

            if early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                return best_test_result
                break

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)
