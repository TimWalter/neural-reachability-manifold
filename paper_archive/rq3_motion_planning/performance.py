import pickle
from pathlib import Path

import torch
from plotly.subplots import make_subplots

import nrm.dataset.se3 as se3
from nrm.visualisation import get_pose_traces
from paper_archive.utils import bootstrap_mean_ci
from nrm.dataset.kinematics import numerical_inverse_kinematics
from nrm.dataset.morphology import sample_morph, get_joint_limits
from nrm.dataset.reachability_manifold import sample_reachable_poses

from paper_archive.rq3_motion_planning.ours import ours
from paper_archive.rq3_motion_planning.baseline import baseline


save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)

num_samples = 10

device = torch.device("cuda")

reachability_list_base = []
deviation_list_base = []
reachability_list_ours = []
deviation_list_ours = []

# Debug
train_loss_list = []
predicted_reachability_list = []
for s in range(1):
    torch.manual_seed(s)
    morph = sample_morph(1, 6, False, device)[0]
    joint_limits = get_joint_limits(morph)
    start = sample_reachable_poses(morph, joint_limits)[0]
    while start.shape[0] == 0:
        start = sample_reachable_poses(morph, joint_limits)[0]
    end = sample_reachable_poses(morph, joint_limits)[0]
    while end.shape[0] == 0:
        end = sample_reachable_poses(morph, joint_limits)[0]

    tangent = se3.log(start, end)
    t = torch.linspace(0, 1, num_samples, device=tangent.device).view(-1, 1)
    target_trajectory = se3.exp(start.repeat(num_samples, 1, 1), t * tangent)

    train_loss, reachability, deviation, debug, trajectory = ours(morph, target_trajectory, 100)
    reachability_list_ours += [torch.tensor(reachability)]
    deviation_list_ours += [torch.tensor(deviation)]

    train_loss, reachability, deviation, _ = baseline(morph, target_trajectory, 100)
    reachability_list_base += [torch.tensor(reachability)]
    deviation_list_base += [torch.tensor(deviation)]

    # Debug
    train_loss_list += [torch.tensor(train_loss)]
    predicted_reachability_list += [torch.tensor(debug["Predicted Reachability"])]

reachability_ours = torch.stack(reachability_list_ours)
deviation_ours = torch.stack(deviation_list_ours)

mean_reachability_ours, lower_reachability_ours, upper_reachability_ours = bootstrap_mean_ci(reachability_ours.numpy())
mean_deviation_ours, lower_deviation_ours, upper_deviation_ours = bootstrap_mean_ci(deviation_ours.numpy())

pickle.dump([mean_reachability_ours, lower_reachability_ours, upper_reachability_ours],
            open(save_dir / "reachability_ours.pkl", "wb"))
pickle.dump([mean_deviation_ours, lower_deviation_ours, upper_deviation_ours],
            open(save_dir / "deviation_ours.pkl", "wb"))

reachability_base = torch.stack(reachability_list_base)
deviation_base = torch.stack(deviation_list_base)

mean_reachability_base, lower_reachability_base, upper_reachability_base = bootstrap_mean_ci(reachability_base.numpy())
mean_deviation_base, lower_deviation_base, upper_deviation_base = bootstrap_mean_ci(deviation_base.numpy())

pickle.dump([mean_reachability_base, lower_reachability_base, upper_reachability_base],
            open(save_dir / "reachability_base.pkl", "wb"))
pickle.dump([mean_deviation_base, lower_deviation_base, upper_deviation_base],
            open(save_dir / "deviation_base.pkl", "wb"))

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",

    "axes.labelsize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 34,
    "axes.titlesize": 34,
    "lines.linewidth": 3,
})

fig, ax = plt.subplots(4, 1, figsize=(15, 20))
colors = sns.color_palette("colorblind", 4)
x = torch.arange(0, mean_reachability_ours.shape[0])
ax[0].plot(mean_reachability_ours, label="Ours")
ax[0].set_ylim(0.0, num_samples)
ax[1].plot(mean_deviation_ours, label="Ours")
ax[2].plot(train_loss_list[0])
ax[3].plot(predicted_reachability_list[0], label="Ours")

ax[0].plot(mean_reachability_base, label="Base")
ax[1].plot(mean_deviation_base, label="Base")

for i in range(len(ax)):
    ax[i].grid(True, linestyle='--', alpha=0.6)

handles, labels = ax[0].get_legend_handles_labels()

# Add the legend to the FIGURE, not the AXES
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.52, 0.0)
)

plt.tight_layout()
plt.show()

fig = make_subplots(
    rows=1, cols=1,
    specs=[[{"type": "scene"}]],
    horizontal_spacing=0.01,
    vertical_spacing=0.05,
)
target_trajectory = target_trajectory.cpu()
trajectory = trajectory.cpu()
morph = morph.cpu()
colors = sns.color_palette("colorblind", n_colors=4 + 1).as_hex()
target_labels = numerical_inverse_kinematics(morph, target_trajectory)[1] != -1
our_labels = numerical_inverse_kinematics(morph, trajectory)[1] != -1
for t in get_pose_traces(morph, target_trajectory[target_labels], colors[0], "reachable target", True):
    fig.add_trace(t, 1, 1)
for t in get_pose_traces(morph, target_trajectory[~target_labels], colors[1], "unreachable target", True):
    fig.add_trace(t, 1, 1)
for t in get_pose_traces(morph, trajectory[our_labels], colors[2], "reachable ours", True):
    fig.add_trace(t, 1, 1)
for t in get_pose_traces(morph, trajectory[~our_labels], colors[3], "unreachable ours", True):
    fig.add_trace(t, 1, 1)

# Clean up Layout
fig.update_layout(
    margin=dict(l=10, r=10, b=10, t=40),  # Minimize outer margins
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.0,
        xanchor="center",
        x=0.5,
        itemsizing='constant',
        groupclick='togglegroup',
        bgcolor='rgba(255,255,255,0.8)'
    ),
    paper_bgcolor='white',
    height=1000,  # Dynamic height
    width=1500,  # Fixed comfortable width
)
axis_style = dict(
    showgrid=True,
    gridcolor='lightgray',  # Subtle grid lines
    gridwidth=1,
    range=[-1, 1],
    showbackground=False,  # Hides the gray walls (cleaner look)
    zeroline=True,  # distinct line at 0
    zerolinecolor='gray',
    showticklabels=True,
    title_font=dict(size=10),
    tickfont=dict(size=8)
)

fig.update_scenes(
    aspectmode='cube',
    xaxis=dict(title='X', **axis_style),
    yaxis=dict(title='Y', **axis_style),
    zaxis=dict(title='Z', **axis_style),
    camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    )
)

fig.show()
