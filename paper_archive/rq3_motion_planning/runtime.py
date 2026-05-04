import torch
import pickle
from pathlib import Path
import nrm.dataset.se3 as se3

from nrm.dataset.morphology import sample_morph, get_joint_limits
from nrm.dataset.reachability_manifold import sample_reachable_poses

from paper_archive.rq3_motion_planning.ours import ours
from paper_archive.rq3_motion_planning.baseline import baseline

from paper_archive.utils import bootstrap_mean_ci
from datetime import datetime

save_dir = Path(__file__).parent / "data"
save_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")
base_runtime = []
ours_runtime = []

sizes = torch.logspace(0,4,10).int()
for size in sizes:
    base_time = []
    ours_time = []
    for seed in range(10):
        torch.manual_seed(seed)
        morph = sample_morph(1, 6, False, device)[0]
        joint_limits = get_joint_limits(morph)
        reachable_poses = sample_reachable_poses(morph.unsqueeze(0).expand(10000, -1, -1),
                                                 joint_limits.unsqueeze(0).expand(10000, -1, -1))[0]
        start = reachable_poses[0]
        end = reachable_poses[1]

        tangent = se3.log(start, end)
        t = torch.linspace(0, 1, size, device=tangent.device).view(-1, 1)
        target_trajectory = se3.exp(start.repeat(size, 1, 1), t * tangent)
        start = datetime.now()
        _ = baseline(morph, target_trajectory, 10, False)
        base_time += [datetime.now() - start]
        start = datetime.now()
        _ = ours(morph, target_trajectory, 10, False)
        ours_time += [datetime.now() - start]
    base_time = torch.tensor([t.seconds *10+ t.microseconds* 10**(-5) for t in base_time])
    ours_time = torch.tensor([t.seconds *10+ t.microseconds* 10**(-5) for t in ours_time])
    base_runtime.append(base_time)
    ours_runtime.append(ours_time)

base_runtime = torch.stack(base_runtime, dim=1)
ours_runtime = torch.stack(ours_runtime, dim=1)

mean_base_runtime, lower_base_runtime, upper_base_runtime = bootstrap_mean_ci(base_runtime.numpy())
mean_ours_runtime, lower_ours_runtime, upper_ours_runtime = bootstrap_mean_ci(ours_runtime.numpy())

pickle.dump([mean_base_runtime, lower_base_runtime, upper_base_runtime], open(save_dir / "base_runtime.pkl", "wb"))
pickle.dump([mean_ours_runtime, lower_ours_runtime, upper_ours_runtime], open(save_dir / "ours_runtime.pkl", "wb"))
