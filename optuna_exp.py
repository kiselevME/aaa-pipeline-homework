from functools import partial
import optuna
import pandas as pd
import src.functions as functions
from train_model import train_lfm

RANDOM_STATE = 1


def objective_with_params(
    trial,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    user2seen: dict,
    user_indes: list,
    node_indes: list,
    optimizer: str,
    experiment_name: str,
    run_name: str,
    K: int
):
    config = {
        'BATCH_SIZE': trial.suggest_int('BATCH_SIZE', 16, 100_000),
        'NUM_NEGATIVES': trial.suggest_int('NUM_NEGATIVES', 1, 20),
        'EDIM': trial.suggest_int('EDIM', 16, 512),
        'EPOCH': trial.suggest_int('EPOCH', 3, 20),
        'OPTIMIZER_NAME': optimizer,
        'LR': trial.suggest_float('LR', 1e-3, 1e1, log=True)
    }

    model = train_lfm(
        config=config,
        df_train=df_train,
        user2seen=user2seen,
        user_indes=user_indes,
        node_indes=node_indes
    )

    hitrate_at_K, precision_at_K = functions.get_metrics(model, df_test, K=K)

    # логруем результат в mlflow
    functions.track_to_mlflow(
        hitrate_at_K=hitrate_at_K,
        precision_at_K=precision_at_K,
        K=K,
        run_name=f'{run_name}-{trial.number}',
        experiment_name=experiment_name,
    )
    return precision_at_K


def optuna_exp(K: int, optimizer: str, experiment_name: str, run_name: str):
    ((df_train, df_test),
     (user_indes, index2user_id),
     (node_indes, index2node)) =\
        functions.prepare_data(
            node2name_path='data/node2name.json',
            dataset_path='data/clickstream.parque'
        )

    user2seen = df_train.groupby('user_index')['node_index'].\
        agg(lambda x: list(set(x)))
    objective = partial(objective_with_params,
                        df_train=df_train,
                        df_test=df_test,
                        user2seen=user2seen,
                        user_indes=user_indes,
                        node_indes=node_indes,
                        optimizer=optimizer,
                        experiment_name=experiment_name,
                        run_name=run_name,
                        K=K)
    # запускаем оптуну
    tpe_sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(sampler=tpe_sampler, direction='maximize')
    study.optimize(objective, n_trials=50)
