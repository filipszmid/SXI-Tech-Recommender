def id_to_name(id, encoder):
    """Convert an ID to a name using the given encoder."""
    return encoder.inverse_transform([id])[0]


def ids_to_names(ids, encoder):
    """Convert a list of IDs to names using the given encoder."""
    return [id_to_name(id, encoder) for id in ids]


def get_current_powers(user_id, df):
    """Retrieve and convert the current powers of a user from IDs to names."""
    if "user_id" not in df.columns:
        raise ValueError(
            "DataFrame does not contain 'user_id'. Check DataFrame preparation steps."
        )

    current_powers = df[df["user_id"] == user_id]["power"].tolist()
    if not current_powers:
        print(f"No powers found for user ID: {user_id}")
        return []
    return current_powers
