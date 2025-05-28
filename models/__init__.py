import torch

from .transformer import GraphTransformer
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN



def xiaorong_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor, leixing: str):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name is BrainNetworkTransformer
    model = leixing
    return eval(model)(config, semantic_similarity_matrix).cuda()


def test_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor, dataset: str):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name is BrainNetworkTransformer
    model = config.model.name + '_abide_test'
    return eval(model)(config, semantic_similarity_matrix).cuda()


def MachineLearning_model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor):
    # model_name is BrainNetworkTransformer
    model = config.jiqixuexi
    return eval(model)(config, semantic_similarity_matrix).cuda()


def model_factory(config: DictConfig, semantic_similarity_matrix: torch.Tensor):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    # model_name is BrainNetworkTransformer
    model = config.model.name + f'_{config.which_model}'
    if config.duibifangfa == 'GAT':
        model = 'GAT'
    if config.duibifangfa == 'MLP':
        model = 'FlattenClassifier'
    if config.duibifangfa == 'BrainGNN':
        model = 'BrainGNN'
    if config.duibifangfa == 'Transformer_duibi':
        model = 'Transformer_duibi'
    if config.duibifangfa == 'Transformer':
        model = 'Transformer'
    if config.duibifangfa == 'FBNETGEN':
        model = 'FBNETGEN'
    if config.duibifangfa == 'BrainNetCNN':
        model = 'BrainNetCNN'
    if config.duibifangfa == 'cc200':
        model = 'BrainNetworkTransformer_abide_cc200'
    if config.duibifangfa == 'BrainNetworkTransformer_abide_meiyouLSTM':
        model = 'BrainNetworkTransformer_abide_meiyouLSTM'

    if config.duibifangfa == 'BrainNetworkTransformer_abide_meiyouTransformer':
        model = 'BrainNetworkTransformer_abide_meiyouTransformer'
    if config.duibifangfa == 'BrainNetworkTransformer_abide_meiyouqunti':
        model = 'BrainNetworkTransformer_abide_meiyouqunti'
    if config.duibifangfa == 'BrainNetworkTransformer_abide_meiyougeti':
        model = 'BrainNetworkTransformer_abide_meiyougeti'
    return eval(model)(config, semantic_similarity_matrix).cuda()
