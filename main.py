import argparse
from baseline import baseline
from optuna_exp import optuna_exp
from optimizer_exp import optimizer_exp
from als_exp import als_exp
from best_model import best_model

K = 30


def main(
    model: str,
    optimizer: str = None,
    experiment_name: str = None,
    run_name: str = None
) -> None:
    if optimizer == '':
        pass

    if model == 'baseline':
        baseline(
            K=K,
            experiment_name=experiment_name,
            run_name=run_name
        )
    elif model == 'lfm' and run_name == 'optuna':
        optuna_exp(
            K=K,
            optimizer=optimizer,
            experiment_name=experiment_name,
            run_name=run_name
        )
    elif model == 'lfm' and run_name == 'optimizer_exp':
        optimizer_exp(
            K=K,
            experiment_name=experiment_name,
            run_name=run_name
        )
    elif model == 'als':
        als_exp(
            K=K,
            experiment_name=experiment_name,
            run_name=run_name
        )
    elif model == 'best_model':
        best_model(
            K=K,
            experiment_name=experiment_name,
            run_name=run_name
        )
    elif model == 'sasrec':
        # to do
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='name of experiment)')
    parser.add_argument('--run_name',
                        help='name of current run in experiment)')
    parser.add_argument('--model', help='name of model)')
    parser.add_argument('--optimizer', help='name of optimizer)')
    args = parser.parse_args()

    experiment_name = args.experiment_name
    run_name = args.run_name
    model = args.model
    optimizer = args.optimizer

    main(
        model=model,
        optimizer=optimizer,
        experiment_name=experiment_name,
        run_name=run_name
    )
