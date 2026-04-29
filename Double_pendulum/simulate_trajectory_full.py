import argparse
import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans']
})

"""
Simulation of a double pendulum with dissipation (viscous damping) at the middle hinge.
State: [theta1, theta2, omega1, omega2]
  theta1, theta2 : absolute angles from vertical (rad)
  omega1, omega2 : angular velocities (rad/s)

Equations from Lagrangian (T-V) plus Rayleigh dissipation function D = 0.5*gamma2*(omega2-omega1)^2.
"""

parser = argparse.ArgumentParser(prog='simulate_trajectory.py',
                                 description='Generates double-pendulum trajectories for a machine learning project.')
parser.add_argument("--num",     default=128,   type=int,   help="number of trajectories to simulate")
parser.add_argument("--points",  default=2048,  type=int,   help="number of time steps per trajectory")
parser.add_argument("--dt",      default=0.005, type=float, help="time step size (s)")
parser.add_argument("--plot",    default=True,  type=bool,  help="plot validation figures")
parser.add_argument("--animate", action='store_true',       help="save a GIF animation")
parser.add_argument("--m1",      default=0.2,   type=float, help="mass of bob 1 (kg)")
parser.add_argument("--m2",      default=0.1,   type=float, help="mass of bob 2 (kg)")
parser.add_argument("--l1",      default=1.0,   type=float, help="length of rod 1 (m)")
parser.add_argument("--l2",      default=1.0,   type=float, help="length of rod 2 (m)")
parser.add_argument("--g",       default=9.8,   type=float, help="gravitational acceleration (m/s^2)")
parser.add_argument("--gamma1",  default=0.01,   type=float, help="damping at the top hinge (N.m.s/rad)")
parser.add_argument("--gamma2",  default=0.05,   type=float, help="damping at the middle hinge (N.m.s/rad)")

args = parser.parse_args()
args.animate = True   # force animation when run without command-line args

def total_energy(theta1, theta2, omega1, omega2):
    """Total mechanical energy E = T + V."""
    T = (0.5 * args.m1 * args.l1**2 * omega1**2
         + 0.5 * args.m2 * (args.l1**2 * omega1**2
                             + args.l2**2 * omega2**2
                             + 2 * args.l1 * args.l2 * omega1 * omega2
                               * np.cos(theta1 - theta2)))
    V = (-(args.m1 + args.m2) * args.g * args.l1 * np.cos(theta1)
         - args.m2 * args.g * args.l2 * np.cos(theta2))
    return T + V

def evolution(state):
    """
    Right-hand side of the ODE: [omega1, omega2, alpha1, alpha2].
    """
    theta1, theta2, omega1, omega2 = state
    d = theta1 - theta2

    M11 = (args.m1 + args.m2) * args.l1**2
    M12 = args.m2 * args.l1 * args.l2 * np.cos(d)
    M22 = args.m2 * args.l2**2
    det = M11 * M22 - M12**2

    rel_vel = omega2 - omega1
    D1 = -args.gamma1 * omega1 + args.gamma2 * rel_vel
    D2 = -args.gamma2 * rel_vel

    f1 = (args.m2 * args.l1 * args.l2 * omega2**2 * np.sin(d)
          - (args.m1 + args.m2) * args.g * args.l1 * np.sin(theta1) + D1)
    f2 = (-args.m2 * args.l1 * args.l2 * omega1**2 * np.sin(d)
          - args.m2 * args.g * args.l2 * np.sin(theta2) + D2)

    alpha1 = (M22 * f1 - M12 * f2) / det
    alpha2 = (M11 * f2 - M12 * f1) / det
    return np.array([omega1, omega2, alpha1, alpha2])

def rk4(f, x, time_step):
    """Classical 4th-order Runge-Kutta."""
    k1 = f(x)
    k2 = f(x + k1 * time_step / 2)
    k3 = f(x + k2 * time_step / 2)
    k4 = f(x + k3 * time_step)
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# Generate trajectories
data = []
np.random.seed(21)

for n in range(args.num):
    theta1_0 = np.random.uniform(-1.0, 1.0)
    theta2_0 = np.random.uniform(-1.0, 1.0)
    x = np.array([theta1_0, theta2_0, 0.0, 0.0])
    time = 0.0
    dataset = []
    for i in range(args.points):
        dataset.append([time] + list(x))
        x = x + rk4(evolution, x, args.dt) * args.dt
        time += args.dt
    data.append(np.array(dataset))
    print(f"{n}/{args.num}", end='\r')

# Save dataset (columns: time, theta1, theta2, omega1, omega2)
os.makedirs("data", exist_ok=True)
dataset_file = "data/dataset.txt"
if os.path.exists(dataset_file):
    os.remove(dataset_file)
for trajectory in data:
    with open(dataset_file, "ab") as f:
        np.savetxt(f, trajectory, delimiter=",")
        f.write(b"\n")
print("\n Done! Trajectories saved into ./data/dataset.txt")

# Validation plots
def validate_trajectory(traj):
    t      = traj[:, 0]
    theta1 = traj[:, 1]
    theta2 = traj[:, 2]
    omega1 = traj[:, 3]
    omega2 = traj[:, 4]
    E = total_energy(theta1, theta2, omega1, omega2)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(t, E, 'k-', lw=1.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Total energy E (J)')
    axes[0].set_title('Energy evolution')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(theta1, theta2, 'b-', lw=0.8, alpha=0.7)
    axes[1].plot(0, 0, 'ro', markersize=6, label='Equilibrium')
    axes[1].set_xlabel('$\\theta_{1}$ (rad)')
    axes[1].set_ylabel('$\\theta_{2}$ (rad)')
    axes[1].set_title('Phase portrait')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/validation_double_pendulum.pdf")
    plt.show()

def animate_trajectory(traj, save=True, speed_multiplier=3.0):
    """Animate the double pendulum motion."""
    t      = traj[:, 0]
    theta1 = traj[:, 1]
    theta2 = traj[:, 2]
    subsample = max(1, int(speed_multiplier / 2))
    t      = t[::subsample]
    theta1 = theta1[::subsample]
    theta2 = theta2[::subsample]
    x1 =  args.l1 * np.sin(theta1)
    y1 = -args.l1 * np.cos(theta1)
    x2 =  x1 + args.l2 * np.sin(theta2)
    y2 =  y1 - args.l2 * np.cos(theta2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-(args.l1 + args.l2 + 0.2), (args.l1 + args.l2 + 0.2))
    ax.set_ylim(-(args.l1 + args.l2 + 0.2), (args.l1 + args.l2 + 0.2))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double pendulum  (middle-hinge damping)')
    line,     = ax.plot([], [], 'o-', lw=2, markersize=8, color='royalblue')
    trace,    = ax.plot([], [], '-',  lw=0.6, alpha=0.4, color='royalblue')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=9)
    trace_x, trace_y = [], []
    def init():
        line.set_data([], [])
        trace.set_data([], [])
        time_text.set_text('')
        return line, trace, time_text
    def update(frame):
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        trace_x.append(x2[frame])
        trace_y.append(y2[frame])
        trace.set_data(trace_x, trace_y)
        time_text.set_text(f't = {t[frame]:.2f} s')
        return line, trace, time_text
    interval_ms = (args.dt * 1000 * subsample) / speed_multiplier
    ani = animation.FuncAnimation(fig, update, frames=len(t),
                                  init_func=init, blit=True,
                                  interval=interval_ms, repeat=False)
    if save:
        os.makedirs("results", exist_ok=True)
        gif_path = "results/double_pendulum_animation.gif"
        try:
            ani.save(gif_path, writer=animation.PillowWriter(fps=30))
            print(f"Animation saved to {gif_path}")
        except Exception as e:
            print(f"Could not save animation: {e}")
            plt.show()
    else:
        plt.show()
    plt.close()

if args.plot:
    idx = np.random.randint(0, args.num)
    validate_trajectory(data[idx])
if args.animate:
    idx = np.random.randint(0, args.num)
    animate_trajectory(data[idx], save=True, speed_multiplier=5.0)