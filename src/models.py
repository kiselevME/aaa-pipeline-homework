import torch
from torch import nn


class LatentFactorModel(nn.Module):
    def __init__(self, edim, user_indexes, node_indexes):
        super(LatentFactorModel, self).__init__()
        self.edim = edim
        self.users = nn.Embedding(max(user_indexes) + 1, edim)
        self.items = nn.Embedding(max(node_indexes) + 1, edim)

    def forward(self, users, items):
        user_embedings = self.users(users).reshape(-1, self.edim)
        item_embedings = self.items(items)
        res = torch.einsum('be,bne->bn', user_embedings, item_embedings)
        return res

    def pred_top_k(self, users, K=10):
        user_embedings = self.users(users).reshape(-1, self.edim)
        item_embedings = self.items.weight
        res = torch.einsum('ue,ie->ui', user_embedings, item_embedings)
        return torch.topk(res, K, dim=1)
