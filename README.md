# scikit-learn in C++

This repository implements various machine learning algorithms from scratch in C++. The purpose is to understand the inner workings of these algorithms by building them from the ground up.

 <!-- DISPLAY=TRUE -->

## Implemented Algorithms

### Linear Regression
- **Simple Linear Regression**: Fits a line to the data points to minimize the mean squared error.
- **Multivariate Linear Regression**: Handles multiple input features to predict the output.

### K-Nearest Neighbors (KNN)
- **KNN for Classification**: Classifies data points based on the majority class among the k-nearest neighbors.
- **KNN for Regression**: Predicts the output based on the average of the k-nearest neighbors.

### Naive Bayes
- **Multinomial Naive Bayes**: Suitable for discrete data, commonly used for text classification.
- **Binary Naive Bayes**: Used for binary classification problems.
- **Gaussian Naive Bayes**: Assumes that the continuous values associated with each class are distributed according to a Gaussian distribution.

### Decision Trees
- **Classification**: Splits the data into subsets based on feature values to create a tree structure, using metrics like Gini index and Information Gain.
- **Regression**: Predicts continuous values by creating a tree where each leaf node represents a mean value of the target variable.

### Random Forests
- **Classification**: An ensemble method that combines multiple decision trees to improve classification performance.
- **Regression**: Uses multiple decision trees to provide more accurate and stable predictions.

### Neural Networks (Implemented in Jupyter Notebook)
- Implements a simple feedforward neural network with backpropagation.

## File Structure

- **Binary-Bayes.cpp**: Implementation of Binary Naive Bayes.
- **Decision_Tree_Regression.cpp**: Implementation of Decision Tree for Regression.
- **Decison_Tree_Classification.cpp**: Implementation of Decision Tree for Classification.
- **Distances.cpp/.h**: Distance metric calculations for KNN.
- **Gaussian-Bayes.cpp**: Implementation of Gaussian Naive Bayes.
- **Gini_index.cpp**: Calculation of Gini index for Decision Trees.
- **Information_gain.cpp**: Calculation of Information Gain for Decision Trees.
- **K-Nearest-Nieghbors.cpp**: Implementation of KNN algorithm.
- **Linear-Regression.cpp**: Implementation of Linear Regression.
- **Multinomial-Bayes.cpp**: Implementation of Multinomial Naive Bayes.
- **Neural-Networks.ipynb**: Implementation of a simple Neural Network.
- **Random_forests.cpp**: Implementation of Random Forest algorithm.
- **Dockerfile**: Docker setup for the project.
- **main.cpp**: Entry point for running the algorithms.

## Dependencies

- **Eigen**: A C++ template library for linear algebra.
- **Boost**: A set of libraries for C++ providing support for various tasks such as linear algebra, pseudorandom number generation, and more.

### Installation

1. Install Eigen:
   ```sh
   sudo apt-get install libeigen3-dev
