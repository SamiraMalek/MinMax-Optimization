import torch
import torch.nn as nn
import numpy as np
import matplotlib.pylab as plt

# Set device to CUDA if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set manual seed for reproducibility
torch.manual_seed(1234)

# Define features and sources
number_feature = 10
number_source = 6  # Should be even

batch_size = 200
data_size = batch_size * 1000

#################################### Data Generation ##################################
# Generate random weights and features
w_variance = 8 * torch.randn(int(number_source / 2), dtype=torch.float64, device=device)
w_star = torch.randn((int(number_source / 2), number_feature), dtype=torch.float64, device=device)
w_star = torch.transpose(w_variance * w_star.T, 0, 1)

x_variance = 8 * torch.randn(int(number_source / 2), dtype=torch.float64, device=device)
x_size = (data_size, number_feature)

# Generate data for each source
for i in range(int(number_source / 2)):
    x_even = x_variance[i] * torch.randn(x_size, dtype=torch.float64, device=device)
    y_even = torch.sign(torch.sign(torch.matmul(x_even, w_star[i])) + 1.0)

    x_odd = x_variance[i] * torch.randn(x_size, dtype=torch.float64, device=device)
    y_odd = torch.sign(torch.sign(torch.matmul(x_odd, w_star[i])) + 1.0)

    locals()[f"x_{2 * i}"] = x_even
    locals()[f"y_{2 * i}"] = y_even
    locals()[f"x_{2 * i + 1}"] = x_odd
    locals()[f"y_{2 * i + 1}"] = y_odd

########################### Helper Functions ######################################
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
def phi_function(x):
    return torch.sqrt(x**2 + 0.5)

# Sigmoid and loss function
sig_function = nn.Sigmoid()
loss_function = nn.BCELoss()

###################################### Optimization ###################################
# Gradient Descent Algorithm (GDA)
def optim_GDA(lr_w, lr_alpha, reference_source, sources, number_source, num_iter, batch_size, data_size, C):
    # Initialize variables
    w = torch.randn(number_feature, dtype=torch.float64, requires_grad=True, device=device)
    alpha = torch.tensor([0.2] * (number_source - 1), dtype=torch.float64, requires_grad=True, device=device)
    alpha = simplex_proj(alpha)

    # Store dynamic values
    alpha_dynamic = torch.zeros((number_source - 1, num_iter), dtype=torch.float64, device=device)
    w_grad_dynamic = torch.zeros(num_iter, dtype=torch.float64, device=device)
    alpha_grad_dynamic = torch.zeros(num_iter, dtype=torch.float64, device=device)

    # Main optimization loop
    for i in range(num_iter):
        # Shuffle data
        if i % int(data_size / batch_size) == 0 and i > 0:
            indices = torch.randperm(data_size)
            for t in range(number_source):
                locals()[f"x_{t}"] = locals()[f"x_{t}"][indices]
                locals()[f"y_{t}"] = locals()[f"y_{t}"][indices]

        k = i % int(data_size / batch_size)
        xT = locals()[f"x_{reference_source}"][k * batch_size:(k + 1) * batch_size, :]
        yT = locals()[f"y_{reference_source}"][k * batch_size:(k + 1) * batch_size]
        fT = loss_function(sig_function(torch.matmul(xT, w)), yT)

        temp = torch.zeros(1, dtype=torch.float64)
        for r, j in zip(sources, range(number_source - 1)):
            x = locals()[f"x_{r}"][k * batch_size:(k + 1) * batch_size, :]
            y = locals()[f"y_{r}"][k * batch_size:(k + 1) * batch_size]
            f = loss_function(sig_function(torch.matmul(x, w)), y)
            temp += alpha[j] * phi_function(fT - f)

        # Objective function
        objective = temp + (C / np.sqrt(batch_size)) * torch.norm(alpha, 2) - 0.001 * torch.norm(w, 2)
        objective.backward(retain_graph=True)

        w_grad_dynamic[i] = torch.norm(w.grad, 2)
        alpha_grad_dynamic[i] = torch.norm(alpha.grad, 2)

        # Gradient updates
        with torch.no_grad():
            w += lr_w * w.grad
            alpha.copy_(simplex_proj(alpha - lr_alpha * alpha.grad))

        # Store dynamic values
        alpha_dynamic[:, i] = alpha
        w.grad.zero_()
        alpha.grad.zero_()

    return alpha_dynamic, w_grad_dynamic, alpha_grad_dynamic

# Run the optimization
num_iter = 400
alpha, w_grad, alpha_grad = optim_GDA(0.005, 0.005, 0, [1, 2, 3, 4, 5], 6, num_iter, batch_size, data_size, 1)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(w_grad.detach().numpy())
axs[0, 0].set_title('|gradF|')

a = torch.zeros((num_iter, 1))
for k in range(num_iter):
    a[k] = torch.sum(w_grad[0:k]) / (k + 1)

axs[0, 1].plot(a)
axs[0, 1].set_title('sum |grad F|/iteration')

axs[1, 0].plot(alpha_grad.detach().numpy())
axs[1, 0].set_title('|grad Alpha|')
axs[1, 0].set(xlabel='Iteration')

for i in range(5):
    axs[1, 1].plot(alpha.detach().numpy()[i, :], label=f"alpha{i+1}")
axs[1, 1].set_title('Alpha')
axs[1, 1].set(xlabel='Iteration')
plt.legend()

plt.show()
