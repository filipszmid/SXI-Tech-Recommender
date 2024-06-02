from tech_recommender.core.recommender import RecommendationWorkflow

if __name__ == "__main__":
    workflow = RecommendationWorkflow(
        model_path="../data/models/trained_model.pth",
        user_encoder_path="../data/models/user_encoder.pkl",
        power_encoder_path="../data/models/power_encoder.pkl",
        data_path="../data/surprise-matrix.csv",
    )

    example_user_id = workflow.df["user_id"].sample(1).iloc[0]
    recommended_powers = workflow.recommend(user_id=example_user_id, verbose=True)
    print("Recommended powers:", recommended_powers)
