# Detecting Deepfake CT Scans with an Ensemble

This project investigates whether an ensemble of pretrained convolutional neural networks can identify manipulated lung CT images while also classifying the scan as benign or malignant. It expands a prior two-model CT-deepfake ensemble into a four-class classifier using **DenseNet-121**, **VGG-16**, and **EfficientNet-B0**.

## Approach

- Ingest DICOM lung CT scans, normalize Hounsfield units, extract image slices, and save class-organized PNGs.
- Apply random horizontal flips and small rotations during training.
- Use frozen ImageNet-pretrained DenseNet-121, VGG-16, and EfficientNet-B0 backbones; flatten and concatenate their features, then train a ReLU/dense/dropout classification head with softmax output.
- Predict four classes: true benign, true malignant, false benign (a real tumor removed), and false malignant (a synthetic tumor inserted).

The data pipeline uses a subset of 1,932 training images and 828 validation images; evaluation is performed on a held-out test set.

## Results

On 852 unseen test images, the ensemble achieved **95.9% accuracy**, **96.9% precision**, and **94.8% recall**. The main failure mode was distinguishing true-malignant from false-benign images, reflecting the visual similarity between genuine tumors and manipulated removals.

## Notebooks

Run in this order:

1. `ingestion_pipeline.ipynb` — DICOM ingestion, normalization, slicing, and dataset preparation.
2. `densenet121_baseline.ipynb` — baseline DenseNet-121 experiment and evaluation.
3. `concatenation.ipynb` — three-model feature-concatenation ensemble and final evaluation.

## Attribution

This work builds on the two-model DenseNet-121 + VGG-16 CT-deepfake ensemble described by Guttikonda, K. K., Cherukumalli, L., Govada, N. V., and Vempati, S. S. (2024), *Identifying Deepfakes in CT Scans of Lung Cancer Using an Ensemble Architecture*, 2024 International Conference on Intelligent Computing and Emerging Communication Technologies (ICEC). https://doi.org/10.1109/ICEC59683.2024.10837331

This repository is an academic research project and is not intended for clinical use.
