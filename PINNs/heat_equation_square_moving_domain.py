import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHA = 0.1

def L_boundary(t):
    """Right boundary position: oscillates between 0.8 and 1.2."""
    return 1.0 + 0.2 * torch.sin(2 * np.pi * t)


class StandardPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def generate_boundary_points(n_points=500):
    points = []
    # Top boundary (y=1, u=1)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    x = torch.rand(n_points, 1, device=device, requires_grad=True) * L_boundary(t)
    y = torch.ones_like(x)
    points.append((torch.cat((x, y, t), dim=1), torch.ones_like(x)))

    # Bottom boundary (y=0, u=0)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    x = torch.rand(n_points, 1, device=device, requires_grad=True) * L_boundary(t)
    y = torch.zeros_like(x)
    points.append((torch.cat((x, y, t), dim=1), torch.zeros_like(x)))

    # Left boundary (x=0, u=0)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    x = torch.zeros_like(y)
    points.append((torch.cat((x, y, t), dim=1), torch.zeros_like(x)))

    # Right boundary (x = L(t), u=0)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    x = L_boundary(t)
    points.append((torch.cat((x, y, t), dim=1), torch.zeros_like(x)))

    # Initial condition (t=0, u=0)
    t = torch.zeros(n_points, 1, device=device, requires_grad=True)
    x = torch.rand(n_points, 1, device=device, requires_grad=True) * L_boundary(t)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat((x, y, t), dim=1), torch.zeros_like(x)))

    return points

def generate_interior_points(n_points=1000):
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    x = torch.rand(n_points, 1, device=device, requires_grad=True) * L_boundary(t)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    return torch.cat((x, y, t), dim=1)

def compute_pde_residual(model, points):
    points.requires_grad_(True)
    u = model(points)

    grad_u = torch.autograd.grad(u.sum(), points, create_graph=True)[0]
    u_t = grad_u[:, 2:3]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    grad_u_x = torch.autograd.grad(u_x.sum(), points, create_graph=True)[0]
    grad_u_y = torch.autograd.grad(u_y.sum(), points, create_graph=True)[0]
    u_xx = grad_u_x[:, 0:1]
    u_yy = grad_u_y[:, 1:2]

    pde_res = u_t - ALPHA * (u_xx + u_yy)
    return pde_res

def train(model, epochs=10001, warmup_epochs=0, loss_csv=None):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)

    loss_history = []
    pde_loss_history = []
    bc_loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        interior_points = generate_interior_points(10000)
        boundary_points = generate_boundary_points(1000)
        interior_points.requires_grad_(True)

        pde_res = compute_pde_residual(model, interior_points)
        pde_loss = torch.mean(pde_res**2)

        bc_loss = 0
        for pts, vals in boundary_points:
            bc_loss += torch.mean((model(pts) - vals) ** 2)

        if epoch < warmup_epochs:
            loss = bc_loss
        else:
            loss = pde_loss + 50 * bc_loss

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())
        pde_loss_history.append(pde_loss.item() if epoch >= warmup_epochs else 0)
        bc_loss_history.append(bc_loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f} "
                  f"(PDE: {pde_loss.item():.6f} | BC: {bc_loss.item():.6f})")

    # Save loss history to CSV
    if loss_csv is not None:
        df_loss = pd.DataFrame({
            'epoch': range(len(loss_history)),
            'loss': loss_history,
            'pde_loss': pde_loss_history,
            'bc_loss': bc_loss_history
        })
        df_loss.to_csv(loss_csv, index=False)
        print(f"Loss history saved to {loss_csv}")

    # Plot training curves
    plt.figure(figsize=(8,4))
    plt.semilogy(loss_history, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return loss_history

def create_solution_plots(model):
    times = np.linspace(0, 1, 6)
    rows, cols = 2, 3
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Heat Equation with Moving Right Boundary (Standard PINN)', fontsize=16)

    levels = np.linspace(0, 1, 11)
    cmap = plt.get_cmap('viridis', len(levels)-1)

    for i, t in enumerate(times):
        row = i // cols
        col = i % cols
        ax = axs[row, col]

        L_val = L_boundary(torch.tensor(t)).item()
        x_phys = np.linspace(0, L_val, 100)
        y_phys = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x_phys, y_phys)

        points = np.array([[xi, yi, t] for yi in y_phys for xi in x_phys])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred = model(points_tensor).cpu().numpy().reshape(len(y_phys), len(x_phys))

        cf = ax.contourf(X, Y, u_pred, levels=levels, cmap=cmap, extend='both')
        ax.set_title(f't = {t:.1f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, L_val)
        ax.set_ylim(0, 1)
        ax.axvline(x=L_val, color='w', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='w', linestyle=':', linewidth=1)

    for j in range(len(times), rows*cols):
        fig.delaxes(axs.flatten()[j])

    cbar = fig.colorbar(cf, ax=axs, fraction=0.02, pad=0.04)
    cbar.set_label('u')
    plt.savefig('moving_domain_solution_standard.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    model = StandardPINN().to(device)
    start = time.time()
    loss_history = train(model, epochs=10001, warmup_epochs=0, loss_csv="loss_history_moving_standard.csv")
    print(f"Training time: {time.time()-start:.2f}s")
    create_solution_plots(model)