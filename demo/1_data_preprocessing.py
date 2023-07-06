import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 255)

# Load data and expand dicts
df = pd.read_json("../data/people-technologies-data.json", orient="records",)
df = df["users"].apply(pd.Series)
df_names = df[["name"]]
df = df[["powers"]]
powers = df["powers"].apply(pd.Series)
df = pd.concat([df.drop(["powers"], axis=1), powers], axis=1)
df.index = df_names

# for col in df.columns:
#     df[col] = df[col].str["name"]

# Rewrite dict structures to power_name//power_level values
for row in range(0, df.shape[0]):
    # print(df.index[row])
    for person_power in range(0, df.shape[1]):
        # print(df.iloc[row, col])
        if str(df.iloc[row, person_power]).lower() != "nan":
            df.iloc[row, person_power] = (
                str(dict(df.iloc[row, person_power])["name"])
                + "//"
                + str(dict(df.iloc[row, person_power])["level"])
            )
df = df.dropna(axis=0, how="all")

# Prepare person_powers_map dict with person : dict_of_powers{name:level}
person_powers_map = {}
unique_powers = []
people = []
for person in range(0, df.shape[0]):
    person_power_dict = {}
    people.append(df.index[person][0])
    for person_power in range(0, df.shape[1]):
        if str(df.iloc[person, person_power]).lower() != "nan":
            power, level = df.iloc[person, person_power].split("//")
            if not power in unique_powers:
                unique_powers.append(power)
            person_power_dict[power] = level
    person_powers_map[df.index[person]] = person_power_dict

# Fill the final matrix with data from dict
final_df = pd.DataFrame(columns=unique_powers, index=people)
for person in final_df.index:
    power_dict = person_powers_map[(person,)]
    for person_power in final_df.columns:
        if person_power in power_dict:
            final_df.at[person, person_power] = power_dict[person_power]

final_df.to_csv(r"../data/matrix.csv", index=True)
