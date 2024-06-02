import os
import json
import random
from typing import Tuple
import numpy as np
import torch
import pandas as pd
import mlflow


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def prepare_data(node2name_path: str, dataset_path: str) -> Tuple:
    with open(node2name_path, 'r') as f:
        node2name = json.load(f)
    node2name = {int(k): v for k, v in node2name.items()}

    df = pd.read_parquet(dataset_path)
    df = df.head(100_000)
    # выделяем train
    df['is_train'] = df['event_date'] <\
        df['event_date'].max() - pd.Timedelta('2 day')
    df['names'] = df['node_id'].map(node2name)

    train_cooks = df[df['is_train']]['cookie_id'].unique()
    train_items = df[df['is_train']]['node_id'].unique()

    df = df[(df['cookie_id'].isin(train_cooks)) &
            (df['node_id'].isin(train_items))]
    # создаем индекс для юзера
    user_indes, index2user_id = pd.factorize(df['cookie_id'])
    df['user_index'] = user_indes
    # создаем индекс для айтема
    node_indes, index2node = pd.factorize(df['node_id'])
    df['node_index'] = node_indes
    # задаем train / test выборку
    df_train, df_test = df[df['is_train']], df[~df['is_train']]
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return ((df_train, df_test),
            (user_indes, index2user_id),
            (node_indes, index2node))


def calc_hitrate(df_preds, K):
    return df_preds[df_preds['rank'] < K].\
        groupby('user_index')['relevant'].max().mean()


def calc_prec(df_preds, K):
    return (df_preds[df_preds['rank'] < K].
            groupby('user_index')['relevant'].mean()).mean()


def get_metrics(model: torch.nn.Module, df_test: pd.DataFrame, K: int = 30):
    test_users = df_test['user_index'].unique()

    preds = model.pred_top_k(torch.tensor(test_users), K)[1].numpy()
    df_preds = pd.DataFrame({
        'node_index': list(preds),
        'user_index': test_users,
        'rank': [[j for j in range(0, K)]for i in range(len(preds))]
    })

    df_preds = df_preds.explode(['node_index', 'rank']).merge(
        df_test[['user_index',
                 'node_index']].assign(relevant=1).drop_duplicates(),
        on=['user_index', 'node_index'],
        how='left'
    )
    df_preds['relevant'] = df_preds['relevant'].fillna(0)

    hitrate = calc_hitrate(df_preds, K)
    prec = calc_prec(df_preds, K)
    return hitrate, prec


def track_to_mlflow(
        hitrate_at_K: str,
        precision_at_K: str,
        K: int,
        run_name: str,
        experiment_name: str = 'homework-pipeline-mekiselev',
        config: dict = None
):
    mlflow.set_tracking_uri('http://84.201.128.89:90/')
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(
            {
                f'hitrate_at_{K}': hitrate_at_K,
                f'preciscion_at_{K}': precision_at_K,
            },
        )
        if config:
            mlflow.log_params(config)
