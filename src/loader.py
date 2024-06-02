import random
import torch
from torch.utils.data import Dataset


class RecDataset(Dataset):
    def __init__(self, users, items, item_per_users):
        self.users = users
        self.items = items
        self.item_per_users = item_per_users

    def __len__(self):
        return len(self.users)

    def __getitem__(self, i):
        user = self.users[i]
        return (torch.tensor(user),
                torch.tensor(self.items[i]),
                self.item_per_users[user])


def collate_fn(batch, num_negatives, num_items):
    users, target_items, users_negatives = [], [], []
    for triplets in batch:
        user, target_item, seen_item = triplets

        users.append(user)
        target_items.append(target_item)
        user_negatives = []

        while len(user_negatives) < num_negatives:
            candidate = random.randint(0, num_items)
            if candidate not in seen_item:
                user_negatives.append(candidate)

        users_negatives.append(user_negatives)

    positive = torch.ones(len(batch), 1)
    negatives = torch.zeros(len(batch), num_negatives)
    labels = torch.hstack([positive, negatives])
    # print(torch.tensor(target_items))
    # print(users_negatives)
    items = torch.hstack([torch.tensor(target_items).reshape(-1, 1),
                          torch.tensor(users_negatives)])
    return torch.hstack(users), items, labels
