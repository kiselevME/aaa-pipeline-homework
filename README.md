# aaa-pipeline-hw

Запуск из консоли (без сборки докер-образа):
- baseline: `python main.py --experiment_name='homework-pipeline-mekiselev' --run_name='baseline' --model='baseline'`
- optuna experiment: `python main.py --experiment_name='homework-pipeline-mekiselev' --run_name='optuna' --model='lfm' --optimizer='Adam'`
- optimizers experiment: `python main.py --experiment_name='homework-pipeline-mekiselev' --run_name='optimizer_exp' --model='lfm'`
- AlternatingLeastSquares experiment: `python main.py --experiment_name='homework-pipeline-mekiselev' --run_name='AlternatingLeastSquares' --model='als'`
- best model: `python main.py --experiment_name='homework-pipeline-mekiselev' --run_name='best_model' --model='best_model'`

Сборка докера и запуск:
1. Создание образа: `docker build -t pipeline:0.0.1 .`
2. Запуск экспериментов:
    - baseline `docker run --name baseline --rm pipeline:0.0.1 --experiment_name='homework-pipeline-mekiselev' --run_name='baseline' --model='baseline'`
    - optuna experiment: `docker run --name optuna --rm pipeline:0.0.1 --experiment_name='homework-pipeline-mekiselev' --run_name='optuna' --model='lfm' --optimizer='Adam'`
    - optimizers experiment: `docker run --name optimizers_exp --rm pipeline:0.0.1 --experiment_name='homework-pipeline-mekiselev' --run_name='optimizer_exp' --model='lfm'`
    - AlternatingLeastSquares experiment: `docker run --name als --rm pipeline:0.0.1 --experiment_name='homework-pipeline-mekiselev' --run_name='AlternatingLeastSquares' --model='als'`
    - best model: `docker run --name best_model --rm pipeline:0.0.1 --experiment_name='homework-pipeline-mekiselev' --run_name='best_model' --model='best_model'`
