def id_to_name(id, encoder):
    """Convert an ID to a name using the given encoder."""
    return encoder.inverse_transform([id])[0]


def ids_to_names(ids, encoder):
    """Convert a list of IDs to names using the given encoder."""
    return [id_to_name(id, encoder) for id in ids]


def get_current_powers(user_id, df, power_encoder):
    """Retrieve and convert the current powers of a user from IDs to names."""
    if "user" not in df.columns:
        raise ValueError(
            "DataFrame does not contain 'user'. Check DataFrame preparation steps."
        )

    current_powers = df[df["user"] == user_id]["power"]
    return ids_to_names(current_powers, power_encoder)
