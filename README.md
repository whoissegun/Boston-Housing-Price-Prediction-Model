# Boston-Housing-Price-Prediction-Model
This repository contains a PyTorch implementation for predicting house prices in the popular Boston Housing dataset (https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset). The model is a simple feedforward neural network trained to minimize the mean squared error (MSE) loss.

Features
The model is a 3-layer feedforward neural network using ReLU activation functions.
The input features are standardized using StandardScaler from sklearn.preprocessing.
The training loop includes early stopping to prevent overfitting.
The model's state dictionary is saved to a .pth file for future use.
Dataset
The dataset used in this project is the Boston Housing dataset, available on Kaggle (link: https://www.kaggle.com/code/prasadperera/the-boston-housing-dataset). It contains 506 instances of data, with 13 features and a target variable (MEDV) representing the median value of owner-occupied homes in $1000's.

Prerequisites:
  Python 3.7 or higher
  PyTorch
  pandas
  scikit-learn
  
 
Model Architecture
  The implemented model is a 3-layer feedforward neural network with the following architecture:

  Input layer: 13 neurons (corresponding to the 13 input features)
  Hidden layer 1: 26 neurons, ReLU activation
  Hidden layer 2: 26 neurons, ReLU activation
  Output layer: 1 neuron (corresponding to the predicted house price)
  Results
  The model's performance is evaluated using mean squared error (MSE) loss on a held-out test set. Early stopping is used to prevent overfitting and to select the best model weights. The model's state dictionary is saved to a file named BostonPricePredictionModel.pth.

  Feel free to explore and experiment with different model architectures, learning rates, or other hyperparameters to improve the model's performance.
