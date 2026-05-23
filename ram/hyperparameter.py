import argparse
import optuna

from train import main

def objective(trial):
    print(f"[TRIAL {trial.number}]")
    kwargs.update({
        "hyperparameter": {
                "dim_encoding": trial.suggest_int("dim_encoding", 16, 1024, step=16),
                "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 4),
                "drop_prob": trial.suggest_float("drop_prob", 0.0, 1.0),
                "dim_decoder": trial.suggest_int("dim_decoder", 128, 2*2048, step=128),
                "num_decoder_layer": trial.suggest_int("num_decoder_layer", 1, 12),

        },
        "trial": trial,
    })
    return main(**kwargs, batch_size=1000, pretrain=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early_stopping", type=int, default=-1)
    args = parser.parse_args()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage="sqlite:///hyperparameter.sqlite3")

    kwargs = vars(args)
    study.optimize(objective, n_trials=100, n_jobs=1)

    print(f"Best value: {study.best_value} (params: {study.best_params})")
