CroIPNet: Cross-Site Individual-Population Graph Integration Network for Neurodevelopmental Disorder Diagnosis

Overview

CroIPNet is a novel deep learning framework designed to enhance the diagnostic accuracy of neurodevelopmental disorders (NDDs) using resting-state fMRI data. The model integrates both individual-specific and population-level brain graphs, utilizing a dual-graph fusion strategy. The framework adapts to the inherent heterogeneity of multi-site fMRI data, improving generalization across different datasets.

Key Features:

- **Cross-Site Graph Integration**: Combines individual brain graphs with population graphs to address site-specific biases in multi-site fMRI data.
- **Multi-Scale Temporal Modeling**: Leverages a hybrid approach using LSTM and Transformer networks to capture short-term and long-term brain dynamics.
- **Contrastive Learning**: Mitigates inter-site variability by aligning representations from different sites.
- **Attention-based Fusion**: A gated attention mechanism dynamically integrates individual and population-level features to improve classification performance.

Installation

Clone the repository and install the required dependencies:

```
git clone https://github.com/Camelus-to/CropIpNet
cd CroIpNet
pip install -r requirements.txt
```

Datasets

The framework has been evaluated on two publicly available resting-state fMRI datasets:

1. **ABIDE I** (Autism Brain Imaging Data Exchange)
2. **ADHD-200** Preprocessing steps include motion correction, spatial smoothing, bandpass filtering, and extraction of mean regional time series based on the AAL or CC200 brain atlases.

Training

To train the model, use the following command:

```
bash

python train.py
```

You can adjust the parameters based on your dataset and requirements.

Hyperparameters:

- `dataset`: Choose from `ABIDE` or `ADHD-200`.
- `atlas`: Choose from `AAL` or `CC200` brain atlases.
- `batch_size`: Batch size for training (default is 32).
- `epochs`: Number of training epochs (default is 50).

Metrics:

- **AUC**
- **Accuracy**
- **Sensitivity**
- **Specificity**

The results for various datasets and models are summarized in the research paper.

References

The full research paper describing the CroIPNet framework, including theoretical background, methods, and results, is available in the repository.

License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
