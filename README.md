# Synthetic Data Experiment with Gradient Descent

This repository contains three Python scripts for conducting experiments on synthetic datasets using gradient descent optimization techniques. The goal is to demonstrate how to generate synthetic data, perform gradient descent with various loss functions, and monitor the performance of model parameters such as weights and gradients over iterations.

## Project Files

- `FullGradientDescentAscent.py`
- `SGDA_IID.py`: Stochastic Gradient Descent Ascent for i.i.D data sources.
- `SGDA_Non_IID.py`:  Stochastic Gradient Descent Ascent for Non i.i.D data sources.

## Contents

### 1. Synthetic Data Generation
Each script generates synthetic data using random values to create input (`x`) and output (`y`) datasets. The input data is randomly generated for `N` sources, and a true weight vector `w_star` is used to define the relationship between `x` and `y`.

### 2. Gradient Descent Optimization
The scripts use gradient descent to minimize a loss function. The following key components are implemented:
- **Binary Cross Entropy Loss**: The `nn.BCELoss` function from PyTorch is used to compute the binary classification error between predictions and true labels.
- **Simplex Projection**: A `simplex_proj()` function ensures that the alpha values remain within a valid range (simplex).
- **Gradient Tracking**: Gradients of both the weights (`w`) and alpha values are tracked and visualized over the iterations.

### 3. Visualization
At the end of each experiment, the scripts generate plots to show:
- The norm of gradients for weights (`|gradF|`).
- The cumulative sum of gradient norms across iterations.
- The evolution of alpha values over time.

## Requirements

To run these scripts, you need the following libraries installed:
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Contributions

Feel free to contribute by submitting pull requests or opening issues to improve the code or extend functionality!
