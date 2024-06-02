import torch
from torch.utils.data import Dataset


class PowerDataset(Dataset):
    def __init__(self, users, powers, levels):
        self.users = users
        self.powers = powers
        self.levels = levels

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.powers[idx], self.levels[idx]
