import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, Adagrad, RMSprop, SGD, ASGD
import src.loader as loader
import src.functions as functions
import src.models as models


RANDOM_STATE = 1


def train_lfm(
    config: dict,
    df_train: pd.DataFrame,
    user2seen: dict,
    user_indes: list,
    node_indes: list
):
    train_dataset = loader.RecDataset(df_train['user_index'].values,
                                      df_train['node_index'],
                                      user2seen)
    functions.set_seed(seed=RANDOM_STATE)
    dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        num_workers=0,
        batch_size=config['BATCH_SIZE'],
        collate_fn=lambda x: loader.collate_fn(
            x,
            config['NUM_NEGATIVES'],
            max(df_train['node_index'].values)
        )
    )

    functions.set_seed(seed=RANDOM_STATE)
    model = models.LatentFactorModel(config['EDIM'], user_indes, node_indes)
    if config['OPTIMIZER_NAME'] == 'Adam':
        optimizer = Adam(model.parameters(), config['LR'])
    elif config['OPTIMIZER_NAME'] == 'AdamW':
        optimizer = AdamW(model.parameters(), config['LR'])
    elif config['OPTIMIZER_NAME'] == 'Adagrad':
        optimizer = Adagrad(model.parameters(), config['LR'])
    elif config['OPTIMIZER_NAME'] == 'RMSprop':
        optimizer = RMSprop(model.parameters(), config['LR'])
    elif config['OPTIMIZER_NAME'] == 'SGD':
        optimizer = SGD(model.parameters(), config['LR'])
    elif config['OPTIMIZER_NAME'] == 'ASGD':
        optimizer = ASGD(model.parameters(), config['LR'])

    for i in range(config['EPOCH']):
        losses = []
        for i in dataloader:
            users, items, labels = i
            optimizer.zero_grad()
            logits = model(users, items)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return model
