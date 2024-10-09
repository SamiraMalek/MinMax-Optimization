import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt

# Experiment on synthetic dataset
number_feature = 10
N = 6

# Set the random seed for reproducibility
torch.manual_seed(123)

# Generate synthetic data
w_star = torch.randn((1, number_feature), dtype=torch.float64)  # 1 * number_feature
batch_size = 200
num_iter = 500
m = batch_size * num_iter

x_size = (m, number_feature)

# Create synthetic input (x) and output (y) for N sources
for i in range(N):
    x = 0.1 * torch.randn(x_size, dtype=torch.float64)
    y = torch.sign(torch.sign(torch.matmul(x, w_star.T)) + 0.5)
    locals()[f"x_{i}"] = x
    locals()[f"y_{i}"] = y

# Invert y_3 values
y_3 = -1 * y_3

# Simplex projection function
def simplex_proj(beta):
    sorted_beta, _ = torch.sort(beta)
    beta_sorted = torch.flip(sorted_beta, [0])
    rho = 1
    for i in range(len(beta) - 1):
        j = len(beta) - i
        test = beta_sorted[j-1] + (1 - torch.sum(beta_sorted[:j])) / j
        if test > 0:
            rho = j
            break

    lam = (1 - torch.sum(beta_sorted[:rho])) / rho
    return torch.maximum(beta + lam, torch.tensor(0.))

# Phi function
def phi_function(x, c):
    return torch.sqrt(x**2 + c)

# Initialize model components
sig_function = nn.Sigmoid()
loss_function = nn.BCELoss()

w = torch.randn(number_feature, 1, dtype=torch.float64, requires_grad=True)
alpha = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64, requires_grad=True)

# Project initial alpha onto simplex
with torch.no_grad():
    alpha.copy_(simplex_proj(alpha))

# Learning rates and constants
lr_w = 0.05
lr_alpha = 0.05
constant1 = 2
constant2 = 1

# Variables to track dynamics over iterations
alpha_dynamic_value = torch.zeros((N-1, num_iter), dtype=torch.float64)
w_grad_dynamic_value = torch.zeros(num_iter, dtype=torch.float64)
alpha_grad_dynamic_value = torch.zeros(num_iter, dtype=torch.float64)

# Indices for the sources used in the experiment
T = torch.tensor([1, 2, 3, 4, 5])

# Optimization loop
for i in range(num_iter):
    indices = torch.randperm(m)
    shuffled_x_0 = x_0[indices]
    shuffled_y_0 = y_0[indices]

    # Calculate loss for reference source
    fT = loss_function(sig_function(torch.matmul(shuffled_x_0, w)), shuffled_y_0)

    temp = torch.zeros(1, dtype=torch.float64)
    
    # Compute the objective function using the sources
    for r, j in zip(T, range(N - 1)):
        x = locals()[f"x_{r}"][indices]
        y = locals()[f"y_{r}"][indices]
        f = loss_function(sig_function(torch.matmul(x, w)), y)
        temp += alpha[j] * phi_function(fT - f, constant2)

    # Full objective function
    object_function = temp + constant1 * torch.norm(alpha / m, 2) - 0.001 * torch.norm(w, 2)

    # Backpropagation
    object_function.backward(retain_graph=True)

    # Track gradients
    w_grad_dynamic_value[i] = torch.norm(w.grad, 2)
    alpha_grad_dynamic_value[i] = torch.norm(alpha.grad, 2)

    # Update weights and alpha
    with torch.no_grad():
        w += lr_w * w.grad
        alpha.copy_(simplex_proj(alpha - lr_alpha * alpha.grad))

    # Store alpha dynamics
    alpha_dynamic_value[:, i] = alpha.T

    # Reset gradients
    w.grad.zero_()
    alpha.grad.zero_()

# Plotting results
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Plot |gradF|
axs[0, 0].plot(w_grad_dynamic_value.detach().numpy())
axs[0, 0].set_title('|gradF|')

# Plot sum of |gradF|/iteration
a = torch.zeros((num_iter, 1))
for k in range(num_iter):
    a[k] = torch.sum(w_grad_dynamic_value[:k]) / (k + 1)

axs[0, 1].plot(a)
axs[0, 1].set_title('Sum |grad F|/iteration')

# Plot |grad Alpha|
axs[1, 0].plot(alpha_grad_dynamic_value.detach().numpy())
axs[1, 0].set_title('|grad Alpha|')
axs[1, 0].set(xlabel='Iteration')

# Plot alpha dynamics
for i in range(5):
    axs[1, 1].plot(alpha_dynamic_value.detach().numpy()[i, :], label=f"alpha{i+1}")
axs[1, 1].set_title('Alpha')
axs[1, 1].set(xlabel='Iteration')
plt.legend()

plt.show()
