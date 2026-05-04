import torch
from pathlib import Path
from nrm.logger import Logger

path = "test_numerical_boundary"

se3_dist = torch.load(Path.cwd() / "data" / path / "se3_dist.pth")
labels = torch.load(Path.cwd() / "data" / path / "labels.pth")

min_unreachable_distance = se3_dist[~labels].min()
max_reachable_distance = se3_dist[labels].max()

if max_reachable_distance > min_unreachable_distance:
    threshold = min_unreachable_distance
    best_f1 = 0.0
    for candidate in torch.linspace(min_unreachable_distance, max_reachable_distance, 100):
        logit = se3_dist < candidate
        f1 = Logger.compute_metrics(logit.float()[:,0], labels)["F1 Score"]
        if f1 > best_f1:
            threshold = candidate
            best_f1 = f1
else:
    threshold = (max_reachable_distance + min_unreachable_distance) / 2

logit = se3_dist < threshold
metrics = Logger.compute_metrics(logit.float()[:,0], labels)
print(threshold)
print(metrics)