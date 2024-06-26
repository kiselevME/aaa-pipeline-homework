from torch.optim import AdamW, Adagrad, RMSprop, SGD, ASGD
import src.functions as functions
from train_model import train_lfm


def optimizer_exp(K: int, experiment_name: str, run_name: str) -> None:
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
    for optimizer in [AdamW, Adagrad, RMSprop, SGD, ASGD]:
        config = {**optuna_best_params, 'OPTIMIZER_NAME': optimizer.__name__}
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
            run_name=f'{run_name}: {optimizer.__name__}',
            experiment_name=experiment_name,
            config=config
        )

        print(f'{optimizer.__name__}:\n\thitrate: {hitrate_at_K}\n\t'
              f'precision: {precision_at_K}', end='\n\n')
