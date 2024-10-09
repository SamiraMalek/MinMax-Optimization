import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt

# Experiment on synthetic dataset
number_feature = 10
N = 6

# Set the random seed
torch.manual_seed(123)

# Generate data x and y
w_star = torch.randn((1, number_feature), dtype=torch.float64)  # 1 * number_feature

batch_size = 200
num_iter = 5000
m = batch_size * num_iter

x_size = (m, number_feature)

# Create synthetic input (x) and output (y) for N sources
for i in range(N):
    x = 0.1 * torch.randn(x_size, dtype=torch.float64)
    y = torch.sign(torch.sign(torch.matmul(x, w_star.T)) + 1.0)
    locals()[f"x_{i}"] = x
    locals()[f"y_{i}"] = y

# Modify y_3 values
y_3 = torch.ones_like(y_3) - y_3

# Simplex projection function
def simplex_proj(beta):
    sorted_beta, _ = torch.sort(beta)
    beta_sorted = torch.flip(sorted_beta, [0])
    rho = 1
    for i in range(len(beta) - 1):
        j = len(beta) - i
        test = beta_sorted[j - 1] + (1 - torch.sum(beta_sorted[:j])) / j
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

# Initialize weights and alpha
w = torch.randn(number_feature, 1, dtype=torch.float64, requires_grad=True)
alpha = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float64, requires_grad=True)

# Project initial alpha onto the simplex
with torch.no_grad():
    alpha.copy_(simplex_proj(alpha))

# Learning rates and constants
lr_w = 0.05
lr_alpha = 0.05
constant1 = 20
constant2 = 1

# Track dynamics over iterations
alpha_dynamic_value = torch.zeros((N-1, num_iter), dtype=torch.float64)
w_grad_dynamic_value = torch.zeros(num_iter, dtype=torch.float64)
alpha_grad_dynamic_value = torch.zeros(num_iter, dtype=torch.float64)

# Indices for the sources used in the experiment
T = torch.tensor([1, 2, 3, 4, 5])

# Optimization loop
for i in range(num_iter):
    # Shuffle and batch data
    xT = x_0[i * batch_size:(i + 1) * batch_size, :]
    yT = y_0[i * batch_size:(i + 1) * batch_size]
    
    # Calculate loss for reference source
    fT = loss_function(sig_function(torch.matmul(xT, w)), yT)

    temp = torch.zeros(1, dtype=torch.float64)
    
    # Compute the objective function using the sources
    for r, j in zip(T, range(N-1)):
        x = locals()[f"x_{r}"][i * batch_size:(i + 1) * batch_size, :]
        y = locals()[f"y_{r}"][i * batch_size:(i + 1) * batch_size]
        f = loss_function(sig_function(torch.matmul(x, w)), y)
        temp += alpha[j] * phi_function(fT - f, constant2)

    # Full objective function
    object_function = temp + (constant1 / np.sqrt(m)) * torch.norm(alpha, 2) - 0.001 * torch.norm(w, 2)

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

# Heatmap for alpha at the last iteration
heat_map = torch.zeros((N, N-1))
heat_map[0, :] = alpha_dynamic_value[:, num_iter - 1]
print(heat_map)
