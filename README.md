# DualCrossNet

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.1-ff6f00.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900.svg)
![Keras](https://img.shields.io/badge/Keras-3.12.0-d00000.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.1-f7931e.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage / Quickstart](#usage--quickstart)
- [Dataset Descriptions](#dataset-descriptions)
- [Code Structure](#code-structure)
- [Citation](#citation)

## Overview

**DualCrossNet** is a dual-stream cross-attention framework designed for antibody–antigen modeling tasks, including neutralization prediction, binding affinity estimation, protein–protein interaction (PPI) analysis, and binding free energy change prediction. 

## Key Features
- **Dual-Stream Cross-Attention**: Utilizes a `LlamaDecoder` block with classical Multi-Head Attention (MHA) followed by bidirectional cross-attention, integrating structural and sequence insights dynamically between antigens and antibodies.
- **Multiple PLM Support**: Built to seamlessly use representations from ProtT5, ESM-2, and SeqVec.
- **Robust Generalization**: Optimized training pipelines (e.g., focal loss, cosine annealing, AdamW) applied across diverse viral and human datasets.
- **Interpretability**: Includes comprehensive interpretability utilities to map and visualize cross-attention scores.

## Architecture

![DualCrossNet Architecture](./image/Architecture-Page-3.drawio.png)

## Biological Plausibility & Interpretability

DualCrossNet goes beyond black-box predictions by incorporating a robust interpretability suite to validate that the model learns genuine *biological grammar* rather than relying on sequence artifacts. 

- **Four-Layer Interpretability**: We employ four independent methods to decode the model's decision-making:
  1. **Cross-Attention (Correlational)**: Maps exactly where the model "looks" when evaluating binding.
  2. **Integrated Gradients (Causal)**: Determines what input features causally drive the final prediction.
  3. **GradCAM (Semi-Causal)**: Filters attention maps by gradient sensitivity to isolate functionally critical residues.
  4. **Chemical Grammar Probing**: Confirms that the model's internal hidden states inherently encode chemical properties (e.g., aromaticity, hydrophobicity) without explicit annotations.

- **HDOCK Cross-Validation**: To ensure biological ground-truth, our interpretability signals are cross-validated against independent, blind molecular docking (HDOCK). 
- **Key Findings**: The model strongly and autonomously recognizes paratope features (like CDR-L1 and CDR-H3) that align perfectly with structural contacts, proving that DualCrossNet intrinsically learns the complex chemical rules governing antibody-antigen binding.

## Installation

### Requirements
* **PyTorch** = 2.4.1
* **TensorFlow** = 2.19.1
* **CUDA** = 12.4
* **GPU** = NVIDIA A100 80GB PCIe (or equivalent)

### Environment Setup

Two separate Conda environments are used to maintain modularity and reproducibility between feature extraction and model training.

#### 1. Feature Extraction Environment

> This environment is used for protein language model–based feature extraction (ProtT5, ESM-2, SeqVec).

```bash
conda env create -f llm_tor2.yaml
conda activate llm_tor2
```

#### 2. Training Environment

> This environment supports deep learning model training, cross-attention modules, and evaluation workflows.

```bash
conda env create -f antibody_dl_environment.yaml
conda activate antibody_dl_environment
```

## Usage / Quickstart

After generating the necessary feature representations, you can utilize the highly optimized training scripts to train the DualCrossNet model. 

```bash
# Example: Training the optimized model on the HIV dataset
conda activate antibody_dl_environment
python script/train_optimized-HIV.py
```

## Dataset Descriptions

#### Ag–Ab Neutralization Datasets (HIV)
The HIV antibody neutralization dataset was sourced from the **HIV Sequence Database**. To ensure dataset diversity and minimize redundancy, antigen–antibody (Ag–Ab) pairs with more than 90% sequence homology in both components were excluded. The final curated benchmark consists of **24,907 neutralizing** and **26,480 non-neutralizing** antibody–antigen pairs, providing a robust large-scale dataset for neutralization prediction. [[Dataset Link]](https://github.com/zhouyu9931/RLEAAI/raw/refs/heads/main/data/dataset_hiv.xlsx)

#### SARS-CoV-2 Neutralization Dataset
Compiled specifically for this study, the SARS-CoV-2 dataset aggregates interaction data from the **Coronavirus Antibody Database (CovAbDab)** and **NCBI**. The dataset was filtered for quality and redundancy to ensure a diverse representation of antibody–antigen interactions. It contains a total of **6,904 interaction pairs**, comprising **3,376 neutralizing (positive)** and **3,528 non-neutralizing (negative)** samples. The data covers multiple viral variants, including **Alpha, Beta, Delta, Gamma, and Omicron**, and is split into a training set (**CoVtr**) of 6,150 pairs and a test set (**CoVtst**) of 754 pairs.

#### Antibody–Antigen Affinity (SAbDab)
The **SAbDab** dataset consists of experimentally resolved antibody–antigen complexes retrieved from the **Protein Data Bank (PDB)**. For this study, the dataset was refined to include only complexes with antigen sequences exceeding 50 residues. Redundancy was further reduced by filtering based on antibody CDR loop similarity, resulting in a high-quality set of **1,513 unique Ag–Ab binding pairs**. [[Dataset Link]](https://raw.githubusercontent.com/gmthu66/AbAgIPA/refs/heads/main/SabDab/SabDabdatabase/positive_StdRecord.csv)

#### Protein–Protein Interaction (PPI) Datasets
- **Human PPI Dataset:** This intra-species dataset includes **36,630 interacting protein pairs** from the **Human Protein Reference Database (HPRD)**. To create a balanced benchmark, **36,480 non-interacting pairs** were generated by pairing proteins from the **LR_PPI** dataset that are localized in different subcellular compartments, minimizing the potential for natural interactions.
- **Yeast PPI Dataset:** The *Saccharomyces cerevisiae* (yeast) core PPI dataset was obtained from the **Database of Interacting Proteins (DIP)** (version 20,070,219). This dataset provides a gold-standard collection of experimentally validated interactions widely used for benchmarking PPI prediction models.

## Code Structure

```text
script/
├── Data-homology-plot.ipynb
├── HIV-RELAAI-SeqVec-FeatureExtraction.ipynb
├── HIV-RLEAAI-Full-FeatureExtraction-ProtT.ipynb
├── LR_PPI-Human-FeatureExtraction-ProtT5.ipynb
├── Negative-Sampling-SabDab.ipynb
├── SabDab2-RELAAI-ESM2-FeatureExtraction.ipynb
├── Yeast-Full-FeatureExtraction-ProtT5.ipynb
├── interp_comprehensive_interpretability_v5.ipynb
├── train_ablation_no_crossattn_Yeast.py
├── train_ablation_singleStream_LLaMA_Yeast.py
├── train_optimized-HIV.py
├── train_optimized-Human.py
├── train_optimized-SabDab.py
├── train_optimized-Yeast.py
└── train_optimized-sars.py
```

> These scripts and notebooks cover feature extraction using multiple protein language models, dataset-specific training pipelines, ablation studies, K-fold cross-validation, and visualization/interpretability utilities.

## Citation

> Citation information will be added here. Please cite the corresponding paper if you use this code or framework.

```bibtex
@article{DualCrossNet2026,
  title   = {A Dual-Stream AI Framework for Multi-Perspective Antibody Functional Landscapes},
  author  = {Delower Hossain, Jake Y Chen},
  journal = {Journal to be added},
  year    = {2026}
}
```
