import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import os
import joblib

from tech_recommender.core.dataset import PowerDataset
from tech_recommender.core.model import RecommenderSystem


class ModelTrainer:
    def __init__(self, data_path, model_save_path):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.user_encoder = LabelEncoder()
        self.power_encoder = LabelEncoder()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df["user"] = self.user_encoder.fit_transform(df["user"])
        df["power"] = self.power_encoder.fit_transform(df["power"])
        train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
        train_dataset = PowerDataset(
            train_df["user"].values, train_df["power"].values, train_df["level"].values
        )
        return DataLoader(train_dataset, batch_size=64, shuffle=True)

    def train_model(self):
        train_loader = self.load_data()
        num_users = len(self.user_encoder.classes_)
        num_powers = len(self.power_encoder.classes_)
        model = RecommenderSystem(num_users, num_powers, emb_size=50)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()

        for epoch in range(10):  # Number of epochs can be parameterized as needed
            total_loss = 0
            for users, powers, levels in train_loader:
                users, powers, levels = users.long(), powers.long(), levels.float()
                optimizer.zero_grad()
                predictions = model(users, powers)
                loss = criterion(predictions, levels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        self.save_model(model)

    def save_model(self, model):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        torch.save(
            model.state_dict(),
            os.path.join(self.model_save_path, "trained_model.pth"),
        )
        joblib.dump(
            self.user_encoder, os.path.join(self.model_save_path, "user_encoder.pkl")
        )
        joblib.dump(
            self.power_encoder, os.path.join(self.model_save_path, "power_encoder.pkl")
        )


if __name__ == "__main__":
    trainer = ModelTrainer("../../data/surprise-matrix.csv", "../../data/models")
    trainer.train_model()
