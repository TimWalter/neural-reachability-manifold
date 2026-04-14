import torch
import matplotlib.pyplot as plt
from datetime import datetime

import nrm.dataset.se3 as se3

from nrm.dataset.reachability_manifold import sample_poses_in_reach, estimate_reachability_manifold
from nrm.dataset.kinematics import inverse_kinematics
from nrm.dataset.morphology import sample_morph

morph = sample_morph(1, 6, False, torch.device("cuda"))[0]

fix_cost = datetime.now()
r_indices = estimate_reachability_manifold(morph.to("cuda"), False, seconds=22)
fix_cost = datetime.now() - fix_cost

x = torch.logspace(0, 8, 9, dtype=torch.int32)

ik_costs = []
fk_costs = []
for num_samples in x:
    cell_indices = se3.index(sample_poses_in_reach(num_samples, morph))
    start = datetime.now()
    _, manipulability = inverse_kinematics(morph.to("cuda"), se3.cell(cell_indices.to("cuda")))
    ground_truth = manipulability != -1
    ik_costs += [datetime.now() - start]
    start = datetime.now()
    labels = torch.isin(cell_indices, r_indices)
    fk_costs += [datetime.now() - start]

fix_cost = torch.tensor([fix_cost.seconds + fix_cost.microseconds * 10**(-6)])
fk_costs =  torch.tensor([fk.seconds + fk.microseconds* 10**(-6) for fk in fk_costs])
ik_costs = torch.tensor([ik.seconds + ik.microseconds * 10**(-6)for ik in ik_costs])

torch.save(fix_cost, "fix_cost.pt")
torch.save(fk_costs, "fk_costs.pt")
torch.save(ik_costs, "ik_costs.pt")

fix_cost = torch.load("fix_cost.pt")
fk_costs = torch.load("fk_costs.pt")
ik_costs = torch.load("ik_costs.pt")



plt.rcParams.update({
    "text.usetex": True,
    "pgf.texsystem": "lualatex",
    "pgf.rcfonts": False,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "pgf.preamble": "\n".join([
        r"\usepackage{fontspec}",
        r"\setmainfont{TeX Gyre Termes}",
        r"\setsansfont{TeX Gyre Heros}",
        r"\setmonofont{TeX Gyre Cursor}",
        r"\usepackage{amsmath}"
    ]),
})

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x, ik_costs/x, label="Numerical IK")
ax.plot(x, (fix_cost + fk_costs)/x, label="Ours")
ax.plot(x, fk_costs/x, label="Ours (var only)")

ax.set_yscale("log")
ax.set_xscale("log")
ax.legend(loc="upper right")
ax.set_xlabel("Number of samples", fontsize=16)
ax.set_ylabel("Time (s)", fontsize=16)

ax.legend(loc='upper left', ncols=2, fontsize=20, fancybox=True,
          frameon=True,
          edgecolor="grey",
          facecolor="white",
          markerscale=2.0)

ax.tick_params(axis='both', which='major', labelsize=14)

plt.grid(True)
plt.savefig("discretisation_runtime.pdf")
plt.show()



