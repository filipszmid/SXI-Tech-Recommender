import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df = pd.read_csv("../data/matrix.csv", index_col=[0])


# %% Dataframe
print(df.head())


# %% Most rare powers - have over 400 nulls
MIN_NULL_NUMBER = 400
rare_powers = df.isna().sum()
rare_powers = rare_powers[rare_powers > MIN_NULL_NUMBER].dropna()
rare_powers = rare_powers.sort_values(axis=0, ascending=False)
print(rare_powers)

# %% Top used powers
NUMBER_OF_POWERS = 20
top_powers = df.notnull().sum()
top_powers = top_powers.sort_values(axis=0, ascending=False)
print(top_powers[:NUMBER_OF_POWERS])


# %% How many powers have a certain person
NUMBER_OF_POPLE = 20
people_powers = df.count(axis=1)
people_powers = people_powers.sort_values(axis=0, ascending=False)
print(people_powers[:NUMBER_OF_POPLE])

# %% What is an average power count number
average_powers_count = people_powers.mean()
print(average_powers_count)

# %% Which persons have less than 10 powers
MIN_POWER_NUMBER = 10
less_powers = people_powers[people_powers < MIN_POWER_NUMBER].dropna()
less_powers = less_powers.sort_values(axis=0, ascending=True)
print(less_powers.head())

# %% All experienced frontend developers (JavaScript greater than 3)
frontend_developers = df.dropna(subset="JavaScript")
experienced_frontend_developers = frontend_developers.query("JavaScript > 2.0")
count_frontend = experienced_frontend_developers[
    experienced_frontend_developers.columns[0]
].count()

print(experienced_frontend_developers[["JavaScript"]])
print(count_frontend)

# %% All experienced backend developers (Python greater than 3)
backend_developers = df.dropna(subset="Python")
experienced_backend_developers = frontend_developers.query("Python > 2.0")
count_backend = experienced_backend_developers[
    experienced_backend_developers.columns[0]
].count()

print(experienced_backend_developers[["Python"]])
print(count_backend)

# %%  Histogram backend vs frontend
experienced_devs = pd.DataFrame(
    {"Number": [count_backend, count_frontend]}, index=["Backend", "Frontend"]
)
print(experienced_devs)
# histogram = experienced_devs.plot.hist(column=["Number"])

#%% Dataframe of experience developers
experienced_devs = pd.concat(
    [experienced_backend_developers, experienced_frontend_developers]
)
print(experienced_devs)
