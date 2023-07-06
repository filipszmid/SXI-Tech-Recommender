# %% Import packages
import random

import datapane as dp
import numpy as np
import pandas as pd
import plotly.express as px
import surprise
from sklearn.manifold import TSNE
from surprise import accuracy
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
df = pd.read_csv("data/surprise-matrix.csv", index_col=[0])

# %% Dataframe:
print(df)
print(df.head())


# %% Dropping powers with count less than minimal
minimal_count = 20
print(
    f"{sum(df.power.value_counts() < minimal_count)} entries occur less than {minimal_count} times in the dataset"
)
df = df.groupby("power").filter(lambda x: len(x) >= minimal_count)

# %% Rewriting dataframe for pairs: (user_id, power_id)
technology_id = {}
ppl_id = {}
id_user = 0
id_power = 0
for row in df.index:
    for col in df.columns:
        if col == "power" and not df.at[row, col] in technology_id:
            print(df.at[row, col])
            technology_id[df.at[row, col]] = id_power
            df.at[row, col] = id_power
            id_power += 1
        if col == "power" and df.at[row, col] in technology_id:
            df.at[row, col] = technology_id[df.at[row, col]]
        if col == "user" and not df.at[row, col] in ppl_id:
            ppl_id[df.at[row, col]] = id_user
            df.at[row, col] = id_user
            id_user += 1
        if col == "user" and df.at[row, col] in ppl_id:
            df.at[row, col] = ppl_id[df.at[row, col]]


df["power"] = df["power"].astype(int)

#%% Important objects
print(technology_id)
print("-----------")
print(ppl_id)
print("-----------")
print(df.head())
print("-----------")
print(df.dtypes)
print("-----------")
print(df["power"][:5])

#%%
reader = surprise.Reader(rating_scale=(1, 4))
data = surprise.Dataset.load_from_df(df[["user", "power", "level"]], reader)


# %% Grid Search

param_grid = {
    "n_epochs": range(650, 800, 50),
    "lr_all": np.arange(
        0.00001, 0.0002, 0.00005
    ),  # The learning rate for all parameters. Default is 0.005
    "reg_all": np.arange(
        0.00001, 0.0002, 0.00005
    ),  # The regularization term for all parameters. Default is 0.02
}
gs = GridSearchCV(
    surprise.SVDpp, param_grid, measures=["rmse", "mae"], cv=3, joblib_verbose=100000000
)
gs.fit(data)

""" 
Experiments:
-SVD 0.8 RMSE
{'n_epochs': 700, 'lr_all': 0.0001, 'reg_all': 0.0001}

-SVDPP 0.65
{'n_epochs': 750, 'lr_all': 0.00016, 'reg_all': 0.00016}
"""

# %% Best RMSE score:
print(gs.best_score["rmse"])

# %% Combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])

# %%
"""https://surprise.readthedocs.io/en/stable/matrix_factorization.html"""

svd = surprise.SVD(verbose=True, n_epochs=1000)  # lr_all=0.0001, reg_all=0.0001
cross = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=3, verbose=True)

# %%
train_set = data.build_full_trainset()
svd.fit(train_set)

# %% How to find person and power ID:
# print(ppl_id["Name Surname"])
print(technology_id["React"])

# %%


def print_indicators(name: str):
    indicators = ["React", "Docker", "Python", "Machine Learning", "HTML5"]
    print(f"Printing indicators for: {name} id: {ppl_id[name]}")
    for i in indicators:
        print(f"{i}: level: {svd.predict(uid=ppl_id[name], iid=technology_id[i]).est}")


# %% Backend vs Frontend
# print_indicators("Bradley Cooper")
# print("------------")
# print_indicators("John Wick")


#%%
def predict_level(user_id, technology_title, model, technology_ids):
    """
    Predicts the level of power for specific user (on a scale of 1-5).
    """

    id = technology_ids[technology_title]
    review_prediction = model.predict(uid=user_id, iid=id)
    return review_prediction.est


# %%
def generate_recommendation(user_id, model, technology_ids, thresh=2.5):
    """
    Generates a power recommendation for a user based on a level threshold. Only
    powers with a predicted rating at or above the threshold will be recommended
    """

    powers = list(technology_ids.keys())
    random.shuffle(powers)

    for power in powers:
        level = predict_level(user_id, power, model, technology_ids)
        if level >= thresh:
            print(f"Recommended power: {power} on level {level}")
            return power


# %% Recommend power for specific persons
id_fe = 1  # ppl_id["John Wick"]
id_po = 2  # ppl_id["Bradley Cooper"]

print("\n Frontend")
generate_recommendation(id_fe, svd, technology_id)
print("\n Product Owner")
generate_recommendation(id_po, svd, technology_id)

#%% Predict power level for a given person.
id_react = technology_id["React"]
print(id_fe)
print(id_react)
print(svd.predict(uid=id_fe, iid=id_react))

# %% Benchmark different algorithms.
"""https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b"""
benchmark = []
# Iterate over all algorithms
for algorithm in [
    surprise.SVD(),
    surprise.SVDpp(),
    surprise.SlopeOne(),
    surprise.NMF(),
    surprise.NormalPredictor(),
    surprise.KNNBaseline(),
    surprise.KNNBasic(),
    surprise.KNNWithMeans(),
    surprise.KNNWithZScore(),
    surprise.BaselineOnly(),
    surprise.CoClustering(),
]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=["RMSE"], cv=3, verbose=False)

    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(
        pd.Series([str(algorithm).split(" ")[0].split(".")[-1]], index=["Algorithm"])
    )
    benchmark.append(tmp)

benchmark = pd.DataFrame(benchmark).set_index("Algorithm").sort_values("test_rmse")

# Accuracy of a certain algorithms
print(benchmark)

# %% Train test split custom size


"""https://surprise.readthedocs.io/en/stable/getting_started.html"""
# sample random train set and test set
# test set is made of 25% of the ratings.
train_set, test_set = train_test_split(data, test_size=0.2)

# We'll use the famous SVDpp algorithm.
svdpp = surprise.SVDpp()


# %%
# Train the algorithm on the train set, and predict ratings for the test set
svdpp.fit(train_set)
predictions = svdpp.test(test_set)

#%% Then compute RMSE
accuracy.rmse(predictions)

# %%
print(train_set)
print(len(test_set))

# %% Train and test an algorithm with the following one-line:
predictions = svdpp.fit(train_set).test(test_set)
print(predictions)

# %% Build model on whole dataset. Retrieve the train set:
train_set = data.build_full_trainset()

# Build an algorithm, and train it.
svdpp = surprise.SVDpp(
    verbose=True, n_factors=20, n_epochs=750, lr_all=0.00016, reg_all=0.00016
)
svdpp.fit(train_set)


# %% Prediction:
pred = svdpp.predict(id_fe, 1, r_ui=4, verbose=True)

# %% Generate recommendations using a new algorithm
generate_recommendation(id_fe, svdpp, technology_id)

# %% Projection, visualization
tsne = TSNE(n_components=2, n_iter=500, verbose=3, random_state=1)
technologies_embedding = tsne.fit_transform(svdpp.qi)  # can be replaced with svd.qi
projection = pd.DataFrame(columns=["x", "y"], data=technologies_embedding)
projection["title"] = technology_id.keys()


# %%
"""    
Some technologies may be generally popular among a wide range of audiences
and thus correspond to points in the center of this scatter plot.
Other powers may fall into very specific persons such as product owner or ceo. 
These powers may correspond to points away from the center of the plot.
"""

fig = px.scatter(projection, x="x", y="y")
fig.show()

# %% Plot points with power titles
def plot_powers(titles, plot_name):
    book_indices = []
    for book in titles:
        book_indices.append(technology_id[book])  # - 1

    power_vector_df = projection.iloc[book_indices]
    fig = px.scatter(power_vector_df, x="x", y="y", text="title",)
    fig.show()

    report = dp.Report(dp.Plot(fig))  # Create a report
    # report.publish(name=plot_name, open=True, visibility="PUBLIC")


# %% There is a possibility to plot chosen technologies or all of them
# technologies = list(power_id.keys())[0:90]
# technologies = list(power_id.keys())[:30]
technologies = list(technology_id.keys())
print(f"Visualizing technologies: {technologies}")
plot_powers(technologies, plot_name="powers_embedding")
