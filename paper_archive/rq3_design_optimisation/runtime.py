import torch
import pickle

import ram.dataset.se3 as se3

from ram.dataset.morphology import sample_morph

from paper_archive.rq3_design_optimisation.ours import ours
from paper_archive.rq3_design_optimisation.baseline import baseline

from paper_archive.utils import bootstrap_mean_ci
from datetime import datetime

device = torch.device("cuda")
base_runtime = []
ours_runtime = []

sizes = torch.logspace(0,4,10).int()
for size in sizes:
    base_time = []
    ours_time = []
    for seed in range(10):
        torch.manual_seed(seed)
        task = se3.random_ball(size, torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.8])).to(device)
        initial_morph = sample_morph(1, 6, False, device)[0]
        start = datetime.now()
        _ = baseline(initial_morph, task, 100, logging=False)
        base_time += [datetime.now() - start]
        start = datetime.now()
        _ = ours(initial_morph, task, 100, logging=False)
        ours_time += [datetime.now() - start]
    base_time = torch.tensor([t.seconds + t.microseconds* 10**(-6) for t in base_time])
    ours_time = torch.tensor([t.seconds + t.microseconds* 10**(-6) for t in ours_time])
    base_runtime.append(base_time)
    ours_runtime.append(ours_time)

base_runtime = torch.stack(base_runtime, dim=1)
ours_runtime = torch.stack(ours_runtime, dim=1)

mean_base_runtime, lower_base_runtime, upper_base_runtime = bootstrap_mean_ci(base_runtime)
mean_ours_runtime, lower_ours_runtime, upper_ours_runtime = bootstrap_mean_ci(ours_runtime)

pickle.dump([mean_base_runtime, lower_base_runtime, upper_base_runtime], open("data/base_runtime.pkl", "wb"))
pickle.dump([mean_ours_runtime, lower_ours_runtime, upper_ours_runtime], open("data/ours_runtime.pkl", "wb"))
