# CSE472 Machine Learning Project Report

## Project Title: Solving Visual Jigsaw Puzzle Using Deep Learning

### 1. Introduction

In this report, we outline our plan to utilize Convolutional Neural Networks (CNNs) for the classification task on the Jigsaw Puzzle dataset. Specifically, we will explore the effectiveness of two prominent CNN architectures: **VGGNet** and **ResNet**. The objective of this project is to develop robust image classifiers capable of accurately categorizing various types of jigsaw puzzle images.

### 2. Dataset Overview

The Jigsaw Puzzle dataset, sourced from Kaggle, comprises a diverse collection of images representing different categories of jigsaw puzzles. The dataset's high-resolution images pose both opportunities and challenges for building accurate classifiers.

- This dataset has 100K images in the form of a Jigsaw puzzle.

- There are two types of puzzle data ----> 2x2 and 3x3.

- The dataset is well-labeled. The label is an array of 4/9 integers, which represent the correct position of the piece in the puzzle.

- Each image has 4/9 pieces which are randomly positioned. The task is to find out the correct positioning of all pieces using neural nets.

### 3. Dataset Exploration

Before proceeding with model development, it is imperative to conduct a comprehensive exploration of the dataset. This involves:

- **Data Distribution Analysis:** Understanding the distribution of images across different puzzle categories to identify potential class imbalances.

- **Data Preprocessing Requirements:** Assessing the need for preprocessing steps such as resizing, normalization, and augmentation to enhance model performance.

- **Sample Image Visualization:** Examining sample images from each category to gain insights into the diversity and complexity of puzzles present in the dataset.

### 4. Model Selection: VGGNet and ResNet

For this project, we have opted to implement a baseline model & two widely recognized CNN architectures:

- **Baseline Model:** Initially, we plan to use a simple multi-layer neural network with a Softmax loss function. This will serve as the baseline against which the performance of more complex models will be compared.

- **VGGNet:** Renowned for its simplicity and effectiveness, VGGNet has demonstrated exceptional performance in various image classification tasks. We plan to leverage a pre-trained VGGNet model and fine-tune it on the Jigsaw Puzzle dataset to exploit its powerful feature extraction capabilities.

- **ResNet:** ResNet, with its innovative residual learning framework, has emerged as a state-of-the-art architecture for deep convolutional networks. We intend to explore the efficacy of ResNet in the context of jigsaw puzzle classification, leveraging its ability to train deeper networks without encountering vanishing gradient problems.

### 5. Project Workflow

Our project workflow encompasses the following key steps:

- **Data Preprocessing:** Resizing, normalization, and augmentation of images to prepare them for model training.

- **Model Development:** Implementation of VGGNet and ResNet architectures using deep learning frameworks such as TensorFlow or PyTorch.

- **Model Training:** Training both VGGNet and ResNet models on the preprocessed dataset, employing techniques such as transfer learning and fine-tuning.

- **Model Evaluation:** Evaluating the trained models on a separate test set to assess their performance metrics, including accuracy, precision, recall, and F1-score.

- **Model Comparison:** Comparing the performance of VGGNet and ResNet models to identify strengths, weaknesses, and areas for improvement.

### 6. Conclusion

This report lays the foundation for our project, outlining our objectives, methodology, and anticipated outcomes. Our project aims to leverage CNN architectures for the classification of jigsaw puzzle images, capitalizing on the impressive performance of CNNs in visual recognition tasks. Through meticulous data exploration, model development, and rigorous evaluation, we aspire to develop accurate and reliable classifiers capable of distinguishing between different types of puzzles with high precision and efficiency.


## Team

- Fabiha Tasneem : 1805072

- Sumaiya Sultana Any : 1805079