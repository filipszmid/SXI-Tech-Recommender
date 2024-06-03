import torch
import joblib
import pandas as pd

from tech_recommender.core.model import RecommenderSystem, MatrixFactorization
from tech_recommender.core.utils import id_to_name, ids_to_names, get_current_powers


class RecommendationWorkflow:
    def __init__(self, model_path, user_encoder_path, power_encoder_path, data_path):
        self.model_path = model_path
        self.user_encoder = joblib.load(user_encoder_path)
        self.power_encoder = joblib.load(power_encoder_path)
        self.df = pd.read_csv(data_path)

        self.df["user_id"] = self.user_encoder.transform(self.df["user"])

        # Load model name from metadata file
        with open("../data/models/best_model_metadata.txt", "r") as file:
            lines = file.readlines()
            model_name_line = next(
                (line for line in lines if line.startswith("Model Name:")), None
            )
            if model_name_line:
                model_type = model_name_line.split(": ")[1].strip()
            else:
                raise ValueError("Model name not found in metadata file.")

        num_users = len(self.user_encoder.classes_)
        num_powers = len(self.power_encoder.classes_)

        if model_type == "RecommenderSystem":
            self.model = RecommenderSystem(num_users, num_powers, emb_size=50)
        elif model_type == "MatrixFactorization":
            self.model = MatrixFactorization(num_users, num_powers)
        else:
            raise ValueError("Unsupported model type specified.")

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def recommend(self, user_id, verbose=False):
        user_id = int(user_id)  # Ensure user_id is an integer
        current_powers = get_current_powers(user_id, self.df)  # Retrieve current powers

        # Get all potential powers as tensor
        powers = torch.tensor(list(range(len(self.power_encoder.classes_))))
        users = torch.tensor([user_id] * len(self.power_encoder.classes_))

        with torch.no_grad():
            predictions = self.model(users, powers).numpy()

        # Sort predictions to get top indices
        sorted_power_indices = predictions.argsort()[::-1]

        if verbose:
            user_name = id_to_name(user_id, self.user_encoder)
            print(f"User: {user_name} ID: {user_id}")
            print(f"User current powers: {current_powers}")

        # Exclude current powers from recommendations
        current_power_ids = set(self.power_encoder.transform(current_powers))
        recommended_ids = []
        for power_id in sorted_power_indices:
            if power_id not in current_power_ids and len(recommended_ids) < 5:
                recommended_ids.append(power_id)
            if len(recommended_ids) == 5:
                break

        recommended_powers_names = ids_to_names(recommended_ids, self.power_encoder)

        if verbose:
            print("Recommended powers (IDs):", recommended_ids)
            print("Recommended powers (Names):", recommended_powers_names)

        return recommended_powers_names
