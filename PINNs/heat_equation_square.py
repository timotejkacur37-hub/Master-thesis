import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib
import os
from PIL import Image

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHA = 0.1


class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def generate_interior_points(n_points=1000):
    x = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    return torch.cat([x, y, t], dim=1)


def generate_boundary_points(n_points=500):
    points = []
    # Bottom (y=0, u=0)
    x = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.zeros(n_points, 1, device=device, requires_grad=True)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat([x, y, t], dim=1), torch.zeros(n_points, 1, device=device)))
    # Left (x=0, u=0)
    x = torch.zeros(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat([x, y, t], dim=1), torch.zeros(n_points, 1, device=device)))
    # Right (x=1, u=0)
    x = torch.ones(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat([x, y, t], dim=1), torch.zeros(n_points, 1, device=device)))
    # Top (y=1, u=1)
    x = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.ones(n_points, 1, device=device, requires_grad=True)
    t = torch.rand(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat([x, y, t], dim=1), torch.ones(n_points, 1, device=device)))
    # Initial (t=0, u=0)
    x = torch.rand(n_points, 1, device=device, requires_grad=True)
    y = torch.rand(n_points, 1, device=device, requires_grad=True)
    t = torch.zeros(n_points, 1, device=device, requires_grad=True)
    points.append((torch.cat([x, y, t], dim=1), torch.zeros(n_points, 1, device=device)))
    return points


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


def train(model, epochs=10001, loss_csv=None):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=500, factor=0.5, verbose=True)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        interior_points = generate_interior_points(10000)
        boundary_points = generate_boundary_points(1000)

        pde_res = compute_pde_residual(model, interior_points)
        pde_loss = torch.mean(pde_res ** 2)

        bc_loss = 0
        for pts, vals in boundary_points:
            bc_loss += torch.mean((model(pts) - vals) ** 2)

        loss = pde_loss + 50.0 * bc_loss

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    # Save loss history to CSV
    if loss_csv is not None:
        import pandas as pd
        df_loss = pd.DataFrame({'epoch': range(len(loss_history)), 'loss': loss_history})
        df_loss.to_csv(loss_csv, index=False)
        print(f"Loss history saved to {loss_csv}")

    return loss_history


def save_pinn_predictions(model, grid_csv, output_csv, device):
    data = np.genfromtxt(grid_csv, delimiter=',', skip_header=1)
    points = data[:, :3]
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = model(points_tensor).cpu().numpy().flatten()
    np.savetxt(output_csv, np.column_stack((points, u_pred)),
               delimiter=',', header='x,y,t,u_pinn', comments='')
    print(f"PINN predictions saved to {output_csv}")

def create_solution_plots(model):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    times = np.linspace(0, 1, 6)

    rows = 2
    cols = 3
    fig, axs = plt.subplots(rows, cols, figsize=(15, 10), constrained_layout=True)
    fig.suptitle('Heat Equation Solution Propagation from Top Boundary', y=1.02)

    for i, t in enumerate(times):
        row = i // cols
        col = i % cols

        points = np.array([[yi, xi, t] for xi in x for yi in y])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred = model(points_tensor).cpu().numpy()

        U = u_pred.reshape(X.shape)

        cmap = plt.get_cmap('viridis', 10)
        levels = np.linspace(0, 1, 11)
        im = axs[row, col].contourf(X, Y, U, levels=levels, cmap=cmap, extend="both")
        axs[row, col].set_title(f't = {t:.1f}')
        axs[row, col].set_xlabel('x')
        axs[row, col].set_ylabel('y')

        axs[row, col].plot([0, 1], [1, 1], 'w--', lw=1)

    # Remove empty subplots
    for i in range(len(times), rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axs[row, col])

    fig.colorbar(im, ax=axs.ravel().tolist()[:len(times)], fraction=0.02, pad=0.04)
    import os
    print("Saving to:", os.getcwd())
    plt.savefig(f"heat_square_plots_no_constrains-epochs=10000.png")
    plt.show()


def create_animation(model):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    os.makedirs('temp_frames', exist_ok=True)
    frame_files = []

    for i, t in enumerate(np.arange(0, 1.01, 0.01)):
        points = np.array([[xi, yi, t] for xi in x for yi in y])
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

        with torch.no_grad():
            u_pred = model(points_tensor).cpu().numpy()

        U = u_pred.reshape(X.shape)

        ax.clear()
        cont = ax.contourf(Y, X, U, levels=20, cmap='viridis', vmin=0, vmax=1)
        ax.plot([0, 1], [1, 1], 'w-', lw=2)
        ax.set_title(f'Time = {t:.2f}')

        frame_file = f'temp_frames/frame_{i:03d}.png'
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        frame_files.append(frame_file)

    images = [Image.open(f) for f in frame_files]
    images[0].save('heat_animation_no_hard_const.gif', save_all=True,
                   append_images=images[1:], duration=100, loop=0)

    # Clean up
    for f in frame_files:
        os.remove(f)
    os.rmdir('temp_frames')

    print("Animation saved as heat_animation_no_hard_const.gif")


if __name__ == "__main__":
    model = PINN().to(device)
    loss_history = train(model, epochs=10001, loss_csv="loss_history_random.csv")

    create_solution_plots(model)

    plt.figure(figsize=(6, 4))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.show()