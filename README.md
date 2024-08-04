# Neural Network Sentiment Analysis

## Project Overview
This project implements a neural network to perform sentiment analysis on book reviews from Amazon. The goal is to classify reviews as positive or negative based on their content.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Experiments and Results](#experiments-and-results)
- [Conclusion](#conclusion)
- [License](#license)

## Dataset
The dataset contains book reviews along with labels indicating whether the review is positive or negative. It is loaded from a CSV file named `bookReviews.csv`.

## Installation
To run this project, you need to have Python installed along with the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

## Model Architecture
The neural network consists of the following layers:

- **Input Layer:** Configured with input shape corresponding to the TF-IDF vectorized features.
- **Hidden Layers:**
  - Four hidden layers with 128, 64, 32, and 16 units respectively, each using the ReLU activation function.
  - Batch Normalization and Dropout layers added for regularization and improved generalization.
- **Output Layer:** A single unit with a sigmoid activation function for binary classification.

## Experiments and Results
Conducted several experiments to optimize the model's performance:

### Experiment 1: Adjusting the Learning Rate
- Tested learning rates: 0.01, 0.001, 0.0001
- Best result: Learning rate of 0.001

### Experiment 2: Adding More Hidden Layers and Adjusting Units
- Configurations: (64, 32), (64, 32, 16), (128, 64, 32, 16)
- Best result: Configuration with four hidden layers (128, 64, 32, 16 units)

### Experiment 3: Tuning TF-IDF Vectorizer
- Configurations: `min_df=1, ngram_range=(1, 1)`, `min_df=2, ngram_range=(1, 1)`, `min_df=1, ngram_range=(1, 2)`
- Best result: `min_df=1, ngram_range=(1, 2)`

### Final Model Performance
- **Test Accuracy:** 0.7932
- The final model configuration provided the highest test accuracy and best generalization.

## Conclusion
The optimal configuration for the sentiment analysis model is:

- **Learning Rate:** 0.001
- **Layer Configuration:** Four hidden layers with 128, 64, 32, and 16 units respectively.
- **Vectorizer Configuration:** `min_df=1, ngram_range=(1, 2)`

This setup achieved the highest test accuracy and provided a good balance between model complexity and performance.

## License
This project is licensed under the MIT License.

