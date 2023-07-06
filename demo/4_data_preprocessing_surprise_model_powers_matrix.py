# %%
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", 255)


# %%
df = pd.read_csv("data/matrix.csv", index_col=[0])
print(df.head())

# %%
# Load data and expand dicts
df = pd.read_json("data/people-technologies-data.json", orient="records",)
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

# %%
print(df)

# %%
print(df.shape)

# %%
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

# %%
print(person_powers_map)

# %%
final_df = pd.DataFrame(columns=["power", "user", "level"])
print(final_df)
print(people)
print(unique_powers)
# %%
for person in people:
    # print(person)
    power_dict = person_powers_map[(person,)]
    for power in unique_powers:
        # print(power)
        if power in power_dict:
            new_row = {"power": power, "user": person, "level": power_dict[power]}
            print(new_row)
            final_df = final_df.append(new_row, ignore_index=True)


# %%
print(final_df)
final_df.to_csv(r"data/surprise-matrix.csv", index=True)
