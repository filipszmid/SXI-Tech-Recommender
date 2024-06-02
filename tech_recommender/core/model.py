import torch
import torch.nn as nn


class RecommenderSystem(nn.Module):
    def __init__(self, num_users, num_powers, emb_size):
        super(RecommenderSystem, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.power_emb = nn.Embedding(num_powers, emb_size)
        # Adjust the input feature size of the linear layer to 2 * emb_size
        self.fc = nn.Linear(2 * emb_size, 1)
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.power_emb.weight.data.uniform_(0, 0.05)

    def forward(self, user, power):
        user_emb = self.user_emb(user)
        power_emb = self.power_emb(power)
        # Concatenate user and power embeddings
        x = torch.cat([user_emb, power_emb], dim=1)
        x = self.fc(x)
        return x.squeeze()
