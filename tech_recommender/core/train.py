import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import joblib
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

from tech_recommender.core.model import RecommenderSystem, MatrixFactorization
from tech_recommender.core.dataset import PowerDataset


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
        return DataLoader(
            PowerDataset(
                train_df["user"].values,
                train_df["power"].values,
                train_df["level"].values,
            ),
            batch_size=64,
            shuffle=True,
        )

    def train_single_model(self, model, train_loader, params):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=params["lr"], weight_decay=params["reg"]
        )
        model.train()
        all_predictions = []
        all_levels = []

        for _ in range(int(params["n_epochs"])):
            total_loss = 0
            for users, powers, levels in train_loader:
                users, powers, levels = users.long(), powers.long(), levels.float()
                optimizer.zero_grad()
                predictions = model(users, powers)
                loss = criterion(predictions, levels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                all_predictions.extend(predictions.detach().numpy())
                all_levels.extend(levels.numpy())

            rmse = np.sqrt(mean_squared_error(all_levels, all_predictions))
            print(f"RMSE: {rmse}")
        return {"loss": rmse, "status": STATUS_OK}

    def objective(self, params):
        train_loader = self.load_data()
        num_users = len(self.user_encoder.classes_)
        num_powers = len(self.power_encoder.classes_)
        model = (
            RecommenderSystem(num_users, num_powers, emb_size=int(params["n_factors"]))
            if params["model_type"] == "RecommenderSystem"
            else MatrixFactorization(
                num_users,
                num_powers,
                emb_size=int(params["n_factors"]),
                init_std=params["init_std"],
            )
        )
        return self.train_single_model(model, train_loader, params)

    def optimize(self):
        space = {
            "model_type": hp.choice(
                "model_type", ["RecommenderSystem", "MatrixFactorization"]
            ),
            "lr": hp.loguniform(
                "lr", np.log(0.00005), np.log(0.005)
            ),  # Finer control over learning rate
            "reg": hp.uniform(
                "reg", 0.0001, 0.05
            ),  # Narrower and lower regularization space
            "n_factors": hp.choice(
                "n_factors", [20, 50, 100, 200]
            ),  # More choices for factors
            "init_std": hp.uniform(
                "init_std", 0.001, 0.05
            ),  # Lower max standard deviation
            "n_epochs": hp.choice(
                "n_epochs", [100, 300, 500, 1000]
            ),  # Higher maximum number of epochs
        }

        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=Trials(),
        )
        return best

    def train_models(self):
        best_params = self.optimize()
        print("Best Parameters:", best_params)

    def save_best_model(self, model, model_name, rmse):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        torch.save(
            model.state_dict(),
            os.path.join(self.model_save_path, "best_trained_model.pth"),
        )
        joblib.dump(
            self.user_encoder, os.path.join(self.model_save_path, "user_encoder.pkl")
        )
        joblib.dump(
            self.power_encoder, os.path.join(self.model_save_path, "power_encoder.pkl")
        )
        with open(
            os.path.join(self.model_save_path, "best_model_metadata.txt"), "w"
        ) as file:
            file.write(f"Model Name: {model_name}\n")
            file.write(f"RMSE: {rmse}\n")


if __name__ == "__main__":
    trainer = ModelTrainer("../../data/surprise-matrix.csv", "../../data/models")
    trainer.train_models()
