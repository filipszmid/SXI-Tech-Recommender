import torch
import torch.nn as nn


class RecommenderSystem(nn.Module):
    def __init__(self, num_users, num_powers, emb_size=50):
        super(RecommenderSystem, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.power_emb = nn.Embedding(num_powers, emb_size)
        self.fc = nn.Linear(2 * emb_size, 1)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.power_emb.weight.data.uniform_(0, 0.05)

    def forward(self, user, power):
        user_emb = self.user_emb(user)
        power_emb = self.power_emb(power)
        x = torch.cat([user_emb, power_emb], dim=1)
        x = self.fc(x)
        return x.squeeze()


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_powers, emb_size=50, init_std=0.01):
        super(MatrixFactorization, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_powers, emb_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_powers, 1)

        # Normal initialization with custom standard deviation
        nn.init.normal_(self.user_emb.weight, mean=0, std=init_std)
        nn.init.normal_(self.item_emb.weight, mean=0, std=init_std)
        nn.init.normal_(self.user_bias.weight, mean=0, std=init_std)
        nn.init.normal_(self.item_bias.weight, mean=0, std=init_std)

    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        ub = self.user_bias(user).squeeze()
        ib = self.item_bias(item).squeeze()
        return (user_emb * item_emb).sum(1) + ub + ib
