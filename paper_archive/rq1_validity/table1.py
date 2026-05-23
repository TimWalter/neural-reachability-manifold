import torch
from tabulate import tabulate

import ram.dataset.r3 as r3
import ram.dataset.so3 as so3
import ram.dataset.se3 as se3

from ram.dataset.morphology import sample_morph, get_joint_limits
from ram.dataset.kinematics import inverse_kinematics
from ram.dataset.workspace import sample_poses_in_reach, fk_approximation, sample_workspace
from ram.logger import binary_confusion_matrix
from paper_archive.utils import bootstrap_mean_ci
torch.manual_seed(1)

# Table 1
for dof in [5,6,7]:
    for level, (interval, n_robots) in enumerate(zip([1, 1, 10, 60], [100, 100, 100, 100])):
        se3.set_level(level + 1)
        print(f"LEVEL {se3.LEVEL}")
        print(f"Fidelity of the discretisation|\t"
              f"# Cells {so3.N_CELLS * (torch.linalg.norm(r3.cell(torch.arange(0, r3.N_CELLS)), dim=1) < 1.0).sum()}|\t"
              f"Distance between neighbouring cells [{se3.MIN_DISTANCE_BETWEEN_CELLS:.3f}, {se3.MAX_DISTANCE_BETWEEN_CELLS:.3f}]\n")

        morphs = sample_morph(n_robots, dof, False,torch.device("cpu"))

        coverage = []
        runtime = []
        benchmarks = []
        metrics = []
        batch_size = None
        for morph_idx, morph in enumerate(morphs):
            num_samples = 100_000
            poses = sample_poses_in_reach(num_samples, morph)
            cell_indices = se3.index(poses)

            _, manipulability = inverse_kinematics(morph.to("cuda"), se3.cell(cell_indices.to(morph.device)).to("cuda"))
            ground_truth = manipulability.cpu() != -1
            if morph.shape[0] < 7 or ground_truth.sum() < num_samples//1:
                subsamples = num_samples // 4
                bmorph = morph.unsqueeze(0).expand(subsamples, -1 ,-1)
                joint_limits = get_joint_limits(morph).unsqueeze(0).expand(subsamples, -1, -1)
                sub_poses, sub_cell_indices = sample_workspace(bmorph, joint_limits)
                subsamples = sub_poses.shape[0]
                poses[:subsamples], cell_indices[:subsamples] = sub_poses.cpu(), sub_cell_indices.cpu()
                ground_truth[:subsamples] = True


            coverage += [ground_truth.sum() / ground_truth.shape[0] * 100]
            runtime += [0]

            true_positives = 0.0
            r_indices = torch.empty(0, dtype=torch.int64)
            while true_positives < 95.0 and runtime[-1] < 600:
                new_r_indices, benchmark, batch_size = fk_approximation(morph.to("cuda"), True, seconds=interval,
                                                                        batch_size=batch_size)
                r_indices = torch.cat([r_indices, new_r_indices]).unique()
                benchmarks += [torch.tensor(benchmark)]
                runtime[-1] += interval

                labels = torch.isin(cell_indices, r_indices)

                ((true_positives, false_negatives),
                 (false_positives, true_negatives)) = binary_confusion_matrix(labels, ground_truth)
            metrics += [[2 * true_positives / (2 * true_positives + false_positives + false_negatives) * 100,
                         true_positives,
                         false_negatives,
                         false_positives,
                         true_negatives]]

        coverage = torch.tensor(coverage)
        runtime = torch.tensor(runtime)
        benchmarks = torch.stack(benchmarks)
        metrics = torch.tensor(metrics)

        # Mean
        mean_coverage = [coverage.mean().item()]
        mean_runtime = [runtime.float().mean().item()]
        mean_benchmark = benchmarks.mean(dim=0).tolist()
        mean_benchmark[0] = int(mean_benchmark[0])
        mean_benchmark[1] = int(mean_benchmark[1])
        mean_metrics = metrics.mean(dim=0).tolist()
        print("MEAN")
        headers = ["Coverage (%)",
                   "Runtime (s)",
                   "Filled Cells / 1s",
                   "Total Samples / 1s",
                   "Efficiency (%)<br>(Total)",
                   "Efficiency (%)<br>(Unique)",
                   "Efficiency (%)<br>(Collision)",
                   "F1 Score (%)",
                   "True Positives (%)",
                   "False Negatives (%)",
                   "False Positives (%)",
                   "True Negatives (%)"]
        print(tabulate([mean_coverage + mean_runtime + mean_benchmark + mean_metrics],
                       headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
        # CI
        mean_coverage, min_coverage, max_coverage = bootstrap_mean_ci(coverage)
        mean_runtime, min_runtime, max_runtime = bootstrap_mean_ci(runtime)
        min_benchmark = []
        max_benchmark = []
        for bench_idx in range(5):
            mean_bench, min_bench, max_bench = bootstrap_mean_ci(benchmarks[:, bench_idx])
            if bench_idx == 0 or bench_idx == 1:
                min_bench = int(min_bench)
                max_bench = int(max_bench)

            min_benchmark += [min_bench]
            max_benchmark += [max_bench]
        min_metrics = []
        max_metrics = []
        for metric_idx in range(6):
            mean_metric, min_metric, max_metric = bootstrap_mean_ci(metrics[:, metric_idx])
            min_metrics += [min_metric]
            max_metrics += [max_metric]
        print("CI Lower")
        print(tabulate([[min_coverage] + [min_runtime] + min_benchmark + min_metrics],
                       headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
        print("CI Upper")
        print(tabulate([[max_coverage] + [max_runtime] + max_benchmark + max_metrics],
                       headers=headers, floatfmt=".4f", intfmt=",", tablefmt="github"))
