# %%
import concurrent.futures
import os
import re

import numpy as np
import pandas as pd

# Plotting plotly
import plotly.express as px
import plotly.figure_factory as ff
import torch
import torch.nn as nn
import torch.nn.functional as F

# BioPython
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import (
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    BertTokenizerFast,
    EsmModel,
)


# %%
def spearman_correlation(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    corr, _ = spearmanr(preds, targets)
    return corr


# %%
model = "./esm_embeddings/esm2_t6_8M_UR50D/"

if os.path.exists(os.path.join(model, "dataset_bio_features.pkl")):
    df = pd.read_pickle(os.path.join(model, "dataset_bio_features.pkl"))

else:
    df = pd.read_pickle(os.path.join(model, "dataset.pkl"))

# %%
df.head()

# %%
# df2 = df.copy()
df = df.dropna(subset=["Pred_affinity"])
df = df[df["Target"] == "MIT_Target"]
df.count()

# %%
# Remove rows where Pred_affinity is NaN
df = df.dropna(subset=["Pred_affinity"])

# %%
df.describe()


# %%
# Compute the BioPython Features
def compute_bio_features(sequence):
    sequence = sequence.upper()

    # Compute the features
    analysis = ProteinAnalysis(sequence)
    features = {
        "molecular_weight": analysis.molecular_weight(),
        "aromaticity": analysis.aromaticity(),
        "isoelectric_point": analysis.isoelectric_point(),
        "helix_fraction": analysis.secondary_structure_fraction()[0],
        "turn_fraction": analysis.secondary_structure_fraction()[1],
        "sheet_fraction": analysis.secondary_structure_fraction()[2],
        "charge_at_pH7": analysis.charge_at_pH(7),
    }

    # Scale the features
    scaler = StandardScaler()
    features = scaler.fit_transform(
        np.array(list(features.values())).reshape(-1, 1)
    ).flatten()

    return features


# %%
# Apply the function to the dataset only if the features are not already computed
feature_names = [
    "molecular_weight",
    "aromaticity",
    "isoelectric_point",
    "helix_fraction",
    "turn_fraction",
    "sheet_fraction",
    "charge_at_pH7",
]

if "molecular_weight" not in df.columns:
    bio_features = df["Sequence"].apply(compute_bio_features)
    df["Sequence_bio_features"] = bio_features
    for i, name in enumerate(feature_names):
        df[name] = df["Sequence_bio_features"].apply(lambda x: x[i])
    df = df.drop(columns=["Sequence_bio_features"])

df.head()

# %%
# # Save the dataframe
# df.to_pickle(os.path.join(model, "dataset_bio_features.pkl"))

# %%
correlation_matrix = df[feature_names + ["Pred_affinity"]].corr()
print(correlation_matrix)

# %%
fig = ff.create_annotated_heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns.tolist(),
    y=correlation_matrix.columns.tolist(),
    annotation_text=correlation_matrix.round(2).values,
    showscale=True,
)
fig.update_layout(
    title="Correlation Matrix", xaxis_title="Features", yaxis_title="Features"
)
fig.show()

# %%
for feature in feature_names:
    fig = px.scatter(
        df,
        x=feature,
        y="Pred_affinity",
        trendline="ols",
        title=f"Relationship between {feature} and Pred_affinity",
    )
    fig.write_image(f"./images/{feature}_vs_Pred_affinity.png")

# %%
# Train a random forest model
embedding_cols = [
    "Sequence_embeddings",
    "HC_embeddings",
    "LC_embeddings",
    "CDRH1_embeddings",
    "CDRH2_embeddings",
    "CDRH3_embeddings",
    "CDRL1_embeddings",
    "CDRL2_embeddings",
    "CDRL3_embeddings",
]


def flatten_embeddings(row):
    embeddings = [row[col].flatten() for col in embedding_cols if col in row]
    all_embeddings = np.concatenate(embeddings)
    bio_features = row[feature_names].values
    return np.concatenate([all_embeddings, bio_features])


# %%
# first, we train only on biopython features
X = df[feature_names].values
y = df["Pred_affinity"].values

# %%
X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7850
)

# %%
X_test.shape

# %%
regressor = RandomForestRegressor(n_estimators=100, random_state=7850, n_jobs=-1)
regressor.fit(X_train, y_train)

# %%
y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
spearman = spearman_correlation(y_pred, y_test)

print(f"MSE: {mse:.2f}, R2: {r2:.2f}, Spearman: {spearman:.2f}")

# %%
# # Now we train on both biopython features and embeddings
# X = df.apply(flatten_embeddings, axis=1).values
# y = df["Pred_affinity"].values

# X.shape, y.shape

# %%
# Now, train a MLP model in PyTorch
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7850
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

X_train_tensor.shape, y_train_tensor.shape

# %%
device = torch.device("mps")


# %%
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_layers=[512, 256, 128, 64],
        dropout_rate=0.2,
    ):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_layers)):
            input_dim = input_dim if i == 0 else hidden_layers[i - 1]
            layers.append(nn.Linear(input_dim, hidden_layers[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output(x)

        return x


model = MLP(
    X_train_tensor.shape[1], hidden_layers=[512, 256, 128, 64], dropout_rate=0.2
)
model.to(device)

# %%
# Early stopping mechanism to prevent overfitting


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"model_checkpoint.pt")
        self.val_loss_min = val_loss


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

early_stopping = EarlyStopping(patience=5, verbose=False)

# Training loop
epochs = 20

with tqdm(total=epochs) as pbar:
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(test_loader)

        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}, Val Loss: {val_loss}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

# %%
# Load the best model
model.load_state_dict(torch.load(f"model_checkpoint.pt"))

# %%
# Evaluate the model
model.eval()
y_pred = []
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        outputs = model(batch_X)
        y_pred.extend(outputs.squeeze().cpu().numpy())

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
spearman = spearman_correlation(y_pred, y_test)

print(f"MSE: {mse:.2f}, R2: {r2:.2f}, Spearman: {spearman:.2f}")

# %%
