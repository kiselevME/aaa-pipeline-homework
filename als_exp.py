import pandas as pd
import implicit
import src.functions as functions

RANDOM_STATE = 1


def matrix_string(list: list, df_train: pd.DataFrame):
    items_string = [0] * df_train['node_index'].nunique()
    for item in list:
        items_string[int(item)] = 1
    return items_string


# AlternatingLeastSquares
def als_exp(K: int, experiment_name: str, run_name: str):
    ((df_train, df_test),
     (user_indes, index2user_id),
     (node_indes, index2node)) =\
        functions.prepare_data(
            node2name_path='data/node2name.json',
            dataset_path='data/clickstream.parque'
        )

    from scipy.sparse import csr_matrix

    train_user_item_data_list = df_train[['user_index', 'node_index']].\
        groupby('user_index').apply(lambda x: matrix_string(x['node_index'],
                                                            df_train)).tolist()
    train_user_item_data = csr_matrix(train_user_item_data_list)

    # initialize a model
    model = implicit.als.AlternatingLeastSquares(
        factors=16,
        iterations=20,
        num_threads=0,
        random_state=RANDOM_STATE
    )
    # train the model on a sparse matrix of user/item/confidence weights
    model.fit(train_user_item_data)

    test_users = df_test['user_index'].unique()

    preds = model.recommend(
        test_users,
        train_user_item_data[test_users].tocsr(),
        N=K
    )[0]
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

    hitrate_at_K = functions.calc_hitrate(df_preds, K)
    precision_at_K = functions.calc_prec(df_preds, K)

    functions.track_to_mlflow(
        hitrate_at_K=hitrate_at_K,
        precision_at_K=precision_at_K,
        K=30,
        run_name=run_name,
        experiment_name=experiment_name,
        config={
                'factors': model.factors,
                'iterations': model.iterations,
                'num_threads': model.num_threads,
                'random_state': model.random_state
            }
    )

    print(hitrate_at_K, precision_at_K)
