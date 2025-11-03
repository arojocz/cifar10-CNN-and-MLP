# CNN vs. MLP: A Comparative Study on CIFAR-10

This repository contains the code and analysis for a project comparing the performance, efficiency, and architectural characteristics of Convolutional Neural Networks (CNNs) versus Multi-Layer Perceptrons (MLPs) on the CIFAR-10 image classification dataset.

This project fulfills the requirements for the "B25 Deep Learning" course.

## Project Overview
The primary goal is to implement and evaluate both CNN and MLP architectures to understand why CNNs are the standard for computer vision tasks. The comparison is based on three key pillars:

Performance: Test Accuracy, and per class Precision, Recall, and F1-Score.

Efficiency: Total trainable parameters and training/test time.

Characteristics: An analysis of translation invariance, number of parameters and learning curves.

## How execute
1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Run any of the training scripts in the src/ folder (both MLP and CNN), for example: python src/CNN_exp1.py; python src/MLP_exp1.py.
4. Once the result files (.pth) are generated, open and run the notebooks/cifar.ipynb notebook to see the comparative analysis.