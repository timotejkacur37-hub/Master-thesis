import os
import argparse
import numpy as np
import torch
from torch import nn
from torch import autograd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 150,
})

parser = argparse.ArgumentParser(description="A pytorch code for learning and testing state space\
                                 trajectory prediction.")

parser.add_argument("--epochs", default=2000, type=int, help="number of epoches for the model to train")
parser.add_argument("--batch_size", default=128, type=int, help="batch size for training of the model")
parser.add_argument("--dt", default=0.005, type=float, help="size of the time step used in the simulation")
parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction,
                    help="do you wish to train a new model?")
parser.add_argument("--log", default=True, action=argparse.BooleanOptionalAction,
                    help="using log loss for plotting and such")
parser.add_argument("--eps", default=5.0, type=float, help="small epsilon used for weights reparametrization")
parser.add_argument("--lbfgs", default=True, action=argparse.BooleanOptionalAction, help="use lbfgs for optimization")
parser.add_argument("--m1", default=0.2, type=float, help="mass of the first pendulum bob (used to convert omega to p)")
parser.add_argument("--m2", default=0.1, type=float, help="mass of the second pendulum bob (used to convert omega to p)")
parser.add_argument("--length", default=1.0, type=float, help="length of the pendulum rods, assumed equal (used to convert omega to p)")
parser.add_argument("--g", default=9.81, type=float, help="gravitational acceleration (used for the potential energy in the total-energy diagnostic plot)")
parser.add_argument("--entropy_input", default="full_state", choices=["full_state", "e_only"],
                     help="'full_state': S(x) may depend on the whole state, degeneracy condition 1 is only "
                          "softly penalized via --lambda_deg1. 'e_only': S depends only on e, which enforces "
                          "degeneracy condition 1 exactly and makes --lambda_deg1 unnecessary.")
parser.add_argument("--lambda_deg1", default=0.025, type=float,
                     help="target weight of the degeneracy-condition-1 penalty (L @ grad_S == 0); ramped up "
                          "from 0 over --deg_warmup_epochs. Only used when --entropy_input full_state.")
parser.add_argument("--lambda_deg2", default=0.01, type=float,
                     help="target weight of the degeneracy-condition-2 penalty (grad_H . dXi/dx* == 0); ramped "
                          "up from 0 over --deg_warmup_epochs.")
parser.add_argument("--deg_warmup_epochs", default=1, type=int,
                     help="number of epochs over which the degeneracy penalty weights are linearly ramped up "
                          "from 0 to their target value, so the network first fits the data before being asked "
                          "to also satisfy the (harder, higher-order-derivative) degeneracy conditions. "
                          "Default (-1) resolves to epochs // 5.")
parser.add_argument("--adam_lr", default=1e-3, type=float, help="learning rate for the Adam optimizer")
parser.add_argument("--grad_clip", default=5, type=float,
                     help="max gradient norm for clipping during training (helps stabilize longer LBFGS runs); set <= 0 to disable")

args, unknown = parser.parse_known_args()

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[])

if args.deg_warmup_epochs < 0:
    args.deg_warmup_epochs = max(1, args.epochs // 5)

# Extracting the data
if os.path.exists("data/dataset.txt") is False:
    raise Exception("We don't have any training data. It should be stored as dataset.txt in the folder data.")

with open("data/dataset.txt", "r", encoding="utf-8") as f:
    data_raw = f.read().strip().split("\n\n")


def compute_total_energy(state, m1, m2, l, g):
    """
    Total (mechanical + internal) energy T + V + e for a state array of shape (..., 5)
    with columns (theta_1, theta_2, p_1, p_2, e), using the same mass matrix M(theta)
    used to convert omega to p, and the standard double-pendulum potential energy for
    two point masses at the end of each rod of length l.
    """
    theta1, theta2 = state[..., 0], state[..., 1]
    p1, p2 = state[..., 2], state[..., 3]
    e = state[..., 4]

    cos_delta = np.cos(theta1 - theta2)
    M11 = (m1 + m2) * l ** 2
    M12 = m2 * l ** 2 * cos_delta
    M22 = m2 * l ** 2
    det = M11 * M22 - M12 ** 2

    T = 0.5 * (M22 * p1 ** 2 - 2 * M12 * p1 * p2 + M11 * p2 ** 2) / det
    V = -(m1 + m2) * g * l * np.cos(theta1) - m2 * g * l * np.cos(theta2)

    return T + V + e


class TrajectoryDataset(Dataset):
    def __init__(self):
        loaded_trajectories = [
            [[float(value) for value in line.split(',')] for line in mat_str.strip().split('\n')]
            for mat_str in data_raw
        ]
        loaded_trajectories = np.array(loaded_trajectories)

        theta1 = loaded_trajectories[:, :, 1]
        theta2 = loaded_trajectories[:, :, 2]
        omega1 = loaded_trajectories[:, :, 3]
        omega2 = loaded_trajectories[:, :, 4]

        m1, m2, l = args.m1, args.m2, args.length
        cos_delta = np.cos(theta1 - theta2)
        M11 = (m1 + m2) * l ** 2
        M12 = m2 * l ** 2 * cos_delta
        M22 = m2 * l ** 2 * np.ones_like(cos_delta)

        p1 = M11 * omega1 + M12 * omega2
        p2 = M12 * omega1 + M22 * omega2

        loaded_trajectories[:, :, 3] = p1
        loaded_trajectories[:, :, 4] = p2

        data = loaded_trajectories[:, 1:-2, :]
        target = loaded_trajectories[:, 2:-1, :]

        global DIMENSION
        DIMENSION = data.shape[2] - 1

        self.position = torch.tensor(data[:, :, 1:], requires_grad=True).float()
        self.target_pos = torch.tensor(target[:, :, 1:], requires_grad=True).float()
        velocity = (data[:, 2:, 1:] - data[:, :-2, 1:]) / (2 * args.dt)

        # boundary conditions
        velocity_first = (data[:, 1:2, 1:] - data[:, 0:1, 1:]) / args.dt
        velocity_last = (data[:, -1:, 1:] - data[:, -2:-1, 1:]) / args.dt
        velocity = np.concatenate([velocity_first, velocity, velocity_last], axis=1)

        self.velocity = torch.tensor(velocity, requires_grad=True).float()
        self.n_samples = self.position.shape[0]

    def __getitem__(self, index):
        return self.position[index], self.target_pos[index], self.velocity[index]

    def __len__(self):
        return self.n_samples


trajectories = TrajectoryDataset()

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

torch.set_default_device(DEVICE)
print(f"Using {DEVICE} for tensor calculations")

generator = torch.Generator(device=DEVICE)
generator.manual_seed(42)

# ## Defining the Model

def rk4(f, x, time_step):
    """
        Classical 4th order Runge Kutta implementation
    """
    k1i = f(x)
    k2i = f(x + k1i * time_step / 2)
    k3i = f(x + k2i * time_step / 2)
    k4i = f(x + k3i * time_step)

    return 1 / 6 * (k1i + 2 * k2i + 2 * k3i + k4i)


class PositiveLinear(nn.Linear):
    """
        A positive layer that we use to enforce convexity and concavity
    """

    def forward(self, input):
        W = self.weight
        eps_tensor = torch.tensor(args.eps, device=W.device, dtype=W.dtype)

        positive_W = W + torch.exp(-eps_tensor)
        negative_W = torch.exp(W - eps_tensor)
        reparam_W = torch.where(W >= 0, positive_W, negative_W)

        return nn.functional.linear(input, reparam_W, self.bias)


class EntropyNetwork(nn.Module):
    """
        For the entropy network we are using a fully input concave neural network achitecture,
        it's a simple alteration of FICNN - fully input convex neural nets.
    """

    def __init__(self):
        super().__init__()
        self.input_dim = 1 if args.entropy_input == "e_only" else DIMENSION

        self.input_layer = nn.Linear(self.input_dim, 8)

        self.prop_layer1 = PositiveLinear(8, 8)
        self.lateral_layer1 = nn.Linear(self.input_dim, 8)

        self.prop_layer2 = PositiveLinear(8, 8)
        self.lateral_layer2 = nn.Linear(self.input_dim, 8)

        self.prop_layer3 = PositiveLinear(8, 8)
        self.lateral_layer3 = nn.Linear(self.input_dim, 8)

        self.prop_layer4 = PositiveLinear(8, 8)
        self.lateral_layer4 = nn.Linear(self.input_dim, 8)

        self.output_layer = PositiveLinear(8, 1)
        self.lateral_layer_out = nn.Linear(self.input_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x0):
        e = x0[..., -1:] if self.input_dim == 1 else x0

        x = nn.Softplus()(self.input_layer(e))

        x = nn.Softplus()(self.prop_layer1(x) + self.lateral_layer1(e))

        x = nn.Softplus()(self.prop_layer2(x) + self.lateral_layer2(e))

        x = nn.Softplus()(self.prop_layer3(x) + self.lateral_layer3(e))

        x = nn.Softplus()(self.prop_layer4(x) + self.lateral_layer4(e))

        S_out = nn.Softplus()(self.output_layer(x) + self.lateral_layer_out(e))

        return -S_out


class DissipationNetwork(nn.Module):
    """
        For this network we are using a more complex architecture to ensure
        only a partial convexity of the output with respect to some inputs.
    """

    def __init__(self):
        super().__init__()
        self.x_input_layer = nn.Linear(DIMENSION, 8)
        self.x_prop_layer1 = nn.Linear(8, 8)

        self.x_lateral_layer_1 = nn.Linear(DIMENSION, 8)
        self.x_lateral_layer_2 = nn.Linear(8, 8)
        self.x_lateral_layer_out = nn.Linear(8, 1)

        self.conjugate_prop_layer_1 = PositiveLinear(8, 8, bias=False)
        self.conjugate_prop_layer_out = PositiveLinear(8, 1, bias=False)

        self.conjugate_prop_layer_1_mid = nn.Linear(8, 8)
        self.conjugate_prop_layer_out_mid = nn.Linear(8, 8)

        self.conjugate_lateral_layer_in = nn.Linear(DIMENSION, 8, bias=False)
        self.conjugate_lateral_layer_1 = nn.Linear(DIMENSION, 8, bias=False)
        self.conjugate_lateral_layer_out = nn.Linear(DIMENSION, 1, bias=False)

        self.conjugate_lateral_layer_in_mid = nn.Linear(DIMENSION, DIMENSION)
        self.conjugate_lateral_layer_1_mid = nn.Linear(8, DIMENSION)
        self.conjugate_lateral_layer_out_mid = nn.Linear(8, DIMENSION)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_raw(self, state, state_conjugate):
        x0 = state
        x0_star = state_conjugate

        x_star = nn.Softplus()(self.x_lateral_layer_1(x0)
                               + self.conjugate_lateral_layer_in(
            torch.mul(x0_star, self.conjugate_lateral_layer_in_mid(x0))))
        x = nn.Softplus()(self.x_input_layer(x0))

        x_star = nn.Softplus()(self.x_lateral_layer_2(x)
                               + self.conjugate_prop_layer_1(
            torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_1_mid(x))))
                               + self.conjugate_lateral_layer_1(
            torch.mul(x0_star, self.conjugate_lateral_layer_1_mid(x))))
        x = nn.Softplus()(self.x_prop_layer1(x))

        Xi_out = nn.Softplus()(self.x_lateral_layer_out(x)
                               + self.conjugate_prop_layer_out(
            torch.mul(x_star, nn.Softplus()(self.conjugate_prop_layer_out_mid(x)))) \
                               + self.conjugate_lateral_layer_out(
            torch.mul(x0_star, self.conjugate_lateral_layer_out_mid(x))))

        return Xi_out

    def forward(self, state, state_conjugate):
        x_star_zeros = torch.zeros_like(state, requires_grad=True)
        Xi_raw = self.forward_raw(state, state_conjugate)
        Xi_at_zero = self.forward_raw(state, x_star_zeros)
        Xi = Xi_raw - Xi_at_zero - (state_conjugate * autograd.grad(Xi_at_zero, x_star_zeros,
                                                                    grad_outputs=torch.ones_like(Xi_at_zero),
                                                                    create_graph=True)[0]).sum(dim=-1).unsqueeze(-1)

        return Xi


class HamiltonianNetwork(nn.Module):
    """
    Standard feedforward network to learn the Energy.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIMENSION, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_normal_(module.weight, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class GENERICDynamics(nn.Module):
    """
    GENERIC model combining Hamiltonian (reversible) and dissipative parts.
    x_dot = J · grad_H(x)  +  dXi/dx*|_{x* = dS/dx}
    """

    def __init__(self):
        super().__init__()
        self.H = HamiltonianNetwork()
        self.S = EntropyNetwork()
        self.Xi = DissipationNetwork()

        L_matrix = torch.zeros(DIMENSION, DIMENSION)
        L_matrix[0, 2] = 1.0
        L_matrix[1, 3] = 1.0
        L_matrix[2, 0] = -1.0
        L_matrix[3, 1] = -1.0
        self.register_buffer("L_matrix", L_matrix)

    def forward(self, x):
        # Reversible part
        H = self.H(x)
        grad_H = autograd.grad(
            H, x, grad_outputs=torch.ones_like(H), create_graph=True
        )[0].float()

        x_dot_H = torch.matmul(grad_H, self.L_matrix.T)

        # Dissipative part
        S = self.S(x)
        x_star = autograd.grad(
            S, x, grad_outputs=torch.ones_like(S), create_graph=True
        )[0].float()

        Xi = self.Xi(x, x_star)
        x_dot_Xi = autograd.grad(
            Xi, x_star, grad_outputs=torch.ones_like(Xi), create_graph=True
        )[0]

        self.degeneracy_residual_1 = torch.matmul(x_star, self.L_matrix.T)
        self.degeneracy_residual_2 = (grad_H * x_dot_Xi).sum(dim=-1, keepdim=True)

        return x_dot_H + x_dot_Xi


# ## Training the Model

L = nn.MSELoss()

if args.train:
    training_trajectories, test_trajectories = random_split(trajectories, [0.8, 0.2], generator=generator)
    model = GENERICDynamics().to(DEVICE)

    val_pos = trajectories.position[test_trajectories.indices].to(DEVICE)
    val_vel = trajectories.velocity[test_trajectories.indices].to(DEVICE)

    lbfgs_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size, shuffle=True,
                                  generator=generator)
    adam_dataloader = DataLoader(dataset=training_trajectories, batch_size=args.batch_size // 2, shuffle=True,
                                 generator=generator)

    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr, amsgrad=True)
    lbfgs_optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-1, max_iter=10, history_size=20,
                                        line_search_fn='strong_wolfe')

    adam_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, mode='min',
                                                                 factor=0.5, patience=15)

    # Training
    losses = []
    val_losses = []
    deg1_losses = []
    deg2_losses = []
    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    def compute_loss(pos, veloc, lambda1, lambda2):

        predicted_veloc = model(pos)
        mse = L(predicted_veloc, veloc)
        deg1 = (model.degeneracy_residual_1 ** 2).mean()
        deg2 = (model.degeneracy_residual_2 ** 2).mean()
        total = mse + lambda1 * deg1 + lambda2 * deg2
        return total, mse, deg1, deg2

    for i in range(args.epochs):
        warmup_frac = min(1.0, (i + 1) / args.deg_warmup_epochs)
        lambda1 = args.lambda_deg1 * warmup_frac if args.entropy_input == "full_state" else 0.0
        lambda2 = args.lambda_deg2 * warmup_frac

        if i < args.epochs - 200 or not args.lbfgs:
            dataloader = adam_dataloader
            optimizer = adam_optimizer
        else:
            dataloader = lbfgs_dataloader
            optimizer = lbfgs_optimizer

        for j, (pos, targ_pos, veloc) in enumerate(dataloader):
            pos = pos.to(DEVICE)
            targ_pos = targ_pos.to(DEVICE)
            veloc = veloc.to(DEVICE)

            if i < args.epochs - 200  or not args.lbfgs:
                optimizer.zero_grad()
                loss, mse, deg1, deg2 = compute_loss(pos, veloc, lambda1, lambda2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            else:
                def closure():
                    optimizer.zero_grad()
                    loss, mse, deg1, deg2 = compute_loss(pos, veloc, lambda1, lambda2)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                    return loss


                optimizer.step(closure)

            loss, mse, deg1, deg2 = compute_loss(pos, veloc, lambda1, lambda2)

        if args.log:
            losses.append(np.log(loss.item()))
        else:
            losses.append(loss.item())
        deg1_losses.append(deg1.item())
        deg2_losses.append(deg2.item())

        val_loss = L(model(val_pos), val_vel).item()
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/model.pth")

        if optimizer is adam_optimizer:
            adam_scheduler.step(val_loss)

        print(f"Epoch {i+1:4d}/{args.epochs} | Loss: {loss.item():.6e} | "
              f"MSE: {mse.item():.6e} | Val MSE: {val_loss:.6e} (best: {best_val_loss:.6e}) | "
              f"Deg1 (raw): {deg1.item():.6e} (w={lambda1:.4f}) | Deg2 (raw): {deg2.item():.6e} (w={lambda2:.4f}) | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(torch.load("models/model.pth", weights_only=True))

else:
    model = GENERICDynamics().to(DEVICE)
    model.load_state_dict(torch.load("models/model.pth", weights_only=True))
    model.eval()

# Evaluation on test set
if args.train:
    test_pos = trajectories.position[test_trajectories.indices].to(DEVICE)
    test_target_pos = trajectories.target_pos[test_trajectories.indices].to(DEVICE)
    test_vel = trajectories.velocity[test_trajectories.indices].to(DEVICE)


else:
    test_pos = trajectories.position.to(DEVICE)
    test_target_pos = trajectories.target_pos.to(DEVICE)
    test_vel = trajectories.velocity.to(DEVICE)

MSE_loss = L(model(test_pos), test_vel).item()
print(f"Loss on the test set is {MSE_loss}.")

os.makedirs("results", exist_ok=True)

# Loss curve
if args.train:
    fig0, ax0 = plt.subplots()
    ax0.set_xlabel("Iterations")
    ax0.set_ylabel("ln(MSE)" if args.log else "MSE")
    ax0.plot(range(len(losses)), losses, label="training loss")
    ax0.legend()
    fig0.savefig("results/loss_2.pdf")

    fig0b, ax0b = plt.subplots()
    ax0b.set_xlabel("Iterations")
    ax0b.set_ylabel("Mean squared residual")
    ax0b.set_yscale("log")
    ax0b.plot(range(len(deg1_losses)), deg1_losses, label=r"Degeneracy 1: $\|L \nabla S\|^2$")
    ax0b.plot(range(len(deg2_losses)), deg2_losses, label=r"Degeneracy 2: $(\nabla H \cdot \partial\Xi/\partial x^*)^2$")
    ax0b.legend()
    fig0b.tight_layout()
    fig0b.savefig("results/degeneracy_training_2.pdf")

# Trajectory comparison
idx = np.random.randint(0, len(test_pos) - 1)
sample = test_pos[idx].cpu().detach().numpy()
time_set = [args.dt * i for i in range(len(sample))]

print("Integrating sample trajectory for plotting...")
prediction = [sample[0]]
for i in tqdm(range(len(sample))):
    state = torch.tensor(prediction[i], dtype=torch.float32, device=DEVICE, requires_grad=True)
    vel = rk4(model, state, args.dt)
    prediction.append(prediction[i] + args.dt * vel.cpu().detach().numpy())
prediction = np.array(prediction)

if DIMENSION == 4:
    var_labels = [r'$\theta_1$ (rad)', r'$\theta_2$ (rad)', r'$\omega_1$ (rad/s)', r'$\omega_2$ (rad/s)']
    fig1, axes = plt.subplots(2, 2, figsize=(10, 6))
    print(f"MSE on test set: {MSE_loss:.3e}")
    for k, (ax, label) in enumerate(zip(axes.flat, var_labels)):
        ax.plot(time_set, sample[:, k], label="original")
        ax.plot(time_set[:-2], prediction[:-3, k], '--', label="prediction")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig("results/trajectory4d_2.pdf")

elif DIMENSION == 5:
    print(f"MSE on test set: {MSE_loss:.3e}")

    var_labels = [r'$\theta_1$ (rad)', r'$\theta_2$ (rad)', r'$p_1$', r'$p_2$']
    fig1, axes = plt.subplots(2, 2, figsize=(10, 6))
    for k, (ax, label) in enumerate(zip(axes.flat, var_labels)):
        ax.plot(time_set, sample[:, k], label="original")
        ax.plot(time_set[:-2], prediction[:-3, k], '--', label="prediction")
        ax.set_xlabel("t (s)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig1.tight_layout()
    fig1.savefig("results/trajectory5d_2.pdf")

    # (b) internal energy e, on its own plot
    fig1e, ax1e = plt.subplots()
    ax1e.plot(time_set, sample[:, 4], label="original")
    ax1e.plot(time_set[:-2], prediction[:-3, 4], '--', label="prediction")
    ax1e.set_xlabel("t (s)")
    ax1e.set_ylabel(r"$e$")
    ax1e.legend()
    fig1e.tight_layout()
    fig1e.savefig("results/internal_energy_2.pdf")

    # (c) total energy T + V + e, on its own plot (should stay ~constant if energy is conserved)
    E_true = compute_total_energy(sample, args.m1, args.m2, args.length, args.g)
    E_pred = compute_total_energy(prediction, args.m1, args.m2, args.length, args.g)
    fig1E, ax1E = plt.subplots()
    ax1E.plot(time_set, E_true, label="original")
    ax1E.plot(time_set[:-2], E_pred[:-3], '--', label="prediction")
    ax1E.set_xlabel("t (s)")
    ax1E.set_ylabel(r"Total energy $T+V+e$")
    ax1E.legend()
    fig1E.tight_layout()
    fig1E.savefig("results/total_energy_2.pdf")

plt.show()