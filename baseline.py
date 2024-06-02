import pandas as pd
import src.functions as functions


def baseline(K: int, experiment_name: str, run_name: str):
    # подготавливаем данные
    ((df_train, df_test), _, _) =\
        functions.prepare_data(
            node2name_path='data/node2name.json',
            dataset_path='data/clickstream.parque'
        )

    # юзеры из теста
    test_users = df_test['user_index'].unique()
    # находим топ-K популярных
    top_popular = df_train[['node_index']].assign(v=1).groupby('node_index').\
        count().reset_index().sort_values(by='v').tail(K)['node_index'].values
    df_preds_top_poplular = pd.DataFrame({
        'node_index': [list(top_popular) for i in test_users],
        'user_index': test_users, 'rank': [[j for j in range(0, K)]
                                           for i in range(len(test_users))]})

    df_preds_top_poplular = df_preds_top_poplular.explode(
        ['node_index', 'rank']
    ).merge(
        df_test[['user_index',
                 'node_index']].assign(relevant=1).drop_duplicates(),
        on=['user_index', 'node_index'],
        how='left',
    )
    df_preds_top_poplular['relevant'] =\
        df_preds_top_poplular['relevant'].fillna(0)

    hitrate_at_K = functions.calc_hitrate(df_preds_top_poplular, K)
    precision_at_K = functions.calc_prec(df_preds_top_poplular, K)

    functions.track_to_mlflow(
            hitrate_at_K=hitrate_at_K,
            precision_at_K=precision_at_K,
            K=K,
            run_name=run_name,
            experiment_name=experiment_name,
        )

    print(hitrate_at_K, precision_at_K)
