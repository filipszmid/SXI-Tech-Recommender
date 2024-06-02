import torch
import joblib
import pandas as pd

from tech_recommender.core.model import RecommenderSystem
from tech_recommender.core.utils import id_to_name, ids_to_names


class RecommendationWorkflow:
    def __init__(self, model_path, user_encoder_path, power_encoder_path, data_path):
        self.model_path = model_path
        self.user_encoder = joblib.load(user_encoder_path)
        self.power_encoder = joblib.load(power_encoder_path)
        self.df = pd.read_csv(data_path)

        self.df["user_id"] = self.user_encoder.transform(self.df["user"])

        num_users = len(self.user_encoder.classes_)
        num_powers = len(self.power_encoder.classes_)
        self.model = RecommenderSystem(num_users, num_powers, emb_size=50)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def recommend(self, user_id, verbose=False):
        user_id = int(user_id)  # Ensure user_id is an integer
        powers = torch.tensor(list(range(len(self.power_encoder.classes_))))
        users = torch.tensor([user_id] * len(self.power_encoder.classes_))
        with torch.no_grad():
            predictions = self.model(users, powers).numpy()
        top_powers = predictions.argsort()[-5:][::-1]

        if verbose:
            print("Top power predictions (IDs):", top_powers)
            user_name = id_to_name(user_id, self.user_encoder)
            print(f"User: {user_name} ID: {user_id}")

        valid_ids = [
            id for id in top_powers if 0 <= id <= len(self.power_encoder.classes_) - 1
        ]
        recommended_powers_names = ids_to_names(valid_ids, self.power_encoder)
        return recommended_powers_names
