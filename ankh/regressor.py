# %%
# %%
import gc
import hashlib
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

torch.cuda.empty_cache()
gc.collect()

# %%
# Seed for reproducibility
seed = 7850
np.random.seed(seed)
random.seed(seed)


# %%
def load_embedding(col, sequence):
    sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
    filepath = f"./embeddings/{col}/{sequence_hash}.npz"
    with np.load(filepath, allow_pickle=True) as data:
        embedding = data["arr_0"]
    return embedding


# %%
def spearman_correlation(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    corr, _ = spearmanr(preds, targets)
    return corr


# %%
# See if pickle file exists

if os.path.exists("./processed_df.pkl"):
    print("Loading data from saved file")
    data = pd.read_pickle("./processed_df.pkl")
    data = data[data["Target"] == "MIT_Target"]
    data = data.sample(frac=0.1, random_state=seed)

else:
    print("Loading data from csv")
    data = pd.read_csv("./data/processed_data1.csv")
    data = data[data["Target"] == "MIT_Target"]
    # Sample only 30% of the data
    data = data.sample(frac=0.3, random_state=seed)

# %%
embedding_columns = [
    "Sequence",
    "HC",
    "LC",
    "CDRH1",
    "CDRH2",
    "CDRH3",
    "CDRL1",
    "CDRL2",
    "CDRL3",
]

# %%
data.head()

# %%
data.describe()

# %%
data["HC_embedding"].iloc[0].shape

# %%
# Load the embeddings from files


# unique_embeddings = {col: {} for col in embedding_columns}

# # Fill the dictionary with unique embeddings
# for col in embedding_columns:
#     unique_sequences = data[col].unique()  # Get unique sequences for the column
#     for sequence in tqdm(unique_sequences, desc=col):
#         if sequence not in unique_embeddings[col]:
#             unique_embeddings[col][sequence] = load_embedding(col, sequence)

# for col in embedding_columns:
#     data[f"{col}_embedding"] = data[col].apply(lambda seq: unique_embeddings[col][seq].flatten())

# # Save the data
# data.to_pickle("./processed_df.pkl")

# %%
data["Sequence_embedding"].shape


# %%
def flatten_embeddings(row):
    flattened_features = []
    for col in embedding_columns:
        col = f"{col}_embedding"
        flat_embedding = (
            row[col].squeeze().flatten()
        )  # Flatten from (1, N, 768) to (N*768,)
        flattened_features.extend(flat_embedding)
    return np.array(flattened_features)


data["features"] = data.apply(flatten_embeddings, axis=1)

# %%
for col in embedding_columns:
    print(f"{col} shape: {data[f'{col}_embedding'].iloc[0].shape}")

# %%
X = np.stack(data["features"].values)
Y = data["Pred_affinity"].values

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=seed
)

# %%
random_forest = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
random_forest.fit(X_train, y_train)

# %%
y_pred = random_forest.predict(X_test)

spearman = spearman_correlation(y_pred, y_test)
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score(y_pred, y_test)

print(f"Spearman Correlation: {spearman}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
