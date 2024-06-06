import src.functions as functions
from train_model import train_lfm


def best_model(K: int, experiment_name: str, run_name: str) -> None:
    # взял из оптуны (лучше подгружать из mlflow, а не хардкодить)
    optuna_best_params = {
        'BATCH_SIZE': 4443,
        'NUM_NEGATIVES': 12,
        'EDIM': 170,
        'EPOCH': 19,
        'LR': 0.028289593099074674
    }

    # подгружаем данные
    ((df_train, df_test),
     (user_indes, index2user_id),
     (node_indes, index2node)) =\
        functions.prepare_data(
            node2name_path='data/node2name.json',
            dataset_path='data/clickstream.parque'
        )
    user2seen = df_train.groupby('user_index')['node_index'].\
        agg(lambda x: list(set(x)))

    # Adam уже есть
    config = {**optuna_best_params, 'OPTIMIZER_NAME': 'Adam'}
    model = train_lfm(
        config=config,
        df_train=df_train,
        user2seen=user2seen,
        user_indes=user_indes,
        node_indes=node_indes
    )
    hitrate_at_K, precision_at_K =\
        functions.get_metrics(model, df_test, K=K)

    functions.track_to_mlflow(
        hitrate_at_K=hitrate_at_K,
        precision_at_K=precision_at_K,
        K=K,
        run_name=run_name,
        experiment_name=experiment_name,
        config=config
    )

    print(f'hitrate: {hitrate_at_K}\n'
          f'precision: {precision_at_K}', end='\n\n')
