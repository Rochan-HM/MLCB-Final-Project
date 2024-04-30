# %% [markdown]
# # Train MLP Models on Embeddings Dataset

# %%
%load_ext autoreload
%autoreload 2

# %%
# Imports
import os
import torch
import esm
import pandas as pd
from tqdm.notebook import tqdm
import lightning as L
from lightning.pytorch import Trainer, seed_everything
from dataset import AntibodyDataset
import requests
import numpy as np
from models import MLPModel, MLPModelWithAttention
import json

# %%
# Set seed
seed = 7850
seed_everything(7850)

# %% [markdown]
# ## Load in ESM 8M dataset

# %%
# Load in ESM 8M dataset
base_path = "../extract_embeddings/data/esm2_650M_UR50D"
dataset = pd.read_pickle(base_path + "/dataset.pkl")

# %%
dataset

# %%
# Add the antigen column to the dataset
antigen_path = base_path + "-antigen.npy"
antigen_embeddings = np.load(antigen_path, allow_pickle=True)

num_rows = dataset.shape[0]
repeated_antigen_embeddings = np.tile(antigen_embeddings, (num_rows, 1))

# %%
dataset['antigen_embeddings'] = repeated_antigen_embeddings.tolist()

# %%
dataset

# %%
dataset.to_pickle(base_path + "/dataset2.pkl")

# %%
models = [
    # "esm2_8M_UR50D", 
    # "esm2_35M_UR50D", 
    # "esm2_150M_UR50D", 
    "esm2_650M_UR50D", 
    "esm2_3B_UR50D",
]

for model in tqdm(models):
    print(f"Loading model: {model}")
    base_path = f"../extract_embeddings/data/{model}"
    dataset = pd.read_pickle(base_path + "/dataset.pkl")

    antigen_path = base_path + "-antigen.npy"
    antigen_embeddings = np.load(antigen_path, allow_pickle=True)

    num_rows = dataset.shape[0]
    repeated_antigen_embeddings = np.tile(antigen_embeddings, (num_rows, 1))
    dataset['antigen_embeddings'] = repeated_antigen_embeddings.tolist()
    dataset.to_pickle(base_path + "/dataset2.pkl")
    print(f"Done model: {model}")


# %% [markdown]
# ## Preprocess dataset

# %%
# Remove all non-target rows
dataset = dataset[dataset['Target'] == 'MIT_Target']
dataset

# %%
# Remove NaN Pred_affinity rows
dataset = dataset[dataset['Pred_affinity'].notna()]
dataset

# %%
# Remove all sequence columns but keep embeddings columns
sequence_columns = ['Sequence', 'HC', 'LC', 'CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3']
dataset = dataset.drop(columns=sequence_columns)

# Remove all columns that don't have embeddings in name and are not Pred_affinity
dataset = dataset.loc[:, dataset.columns.str.contains('embeddings|Pred_affinity')]
dataset

# %% [markdown]
# ## Split dataset into train, val, and test

# %%
# Create train, validation, and test datasets using a 70/15/15 split
train_dataset = dataset.sample(frac=0.7)
dataset = dataset.drop(train_dataset.index)
validation_dataset = dataset.sample(frac=0.5)
test_dataset = dataset.drop(validation_dataset.index)

# %%
# Print length of each dataset
print(f"Length of train dataset: {len(train_dataset)}")
print(f"Length of validation dataset: {len(validation_dataset)}")
print(f"Length of test dataset: {len(test_dataset)}")

# %%
train_dataset = AntibodyDataset(train_dataset, concat_embeddings_into_1=False)
validation_dataset = AntibodyDataset(validation_dataset, concat_embeddings_into_1=False)
test_dataset = AntibodyDataset(test_dataset, concat_embeddings_into_1=False)

# %% [markdown]
# ## Train Model

# %%
# Create dataloaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

# %%
# Train model using PyTorch Lightning
# model = MLPModel(
#     input_dim=train_dataset[0][0].shape[0],
#     dropout=0.1,
#     esm_model='8M',
# )
model = MLPModelWithAttention(
    num_sequences = 9,
    embedding_dim = 320
)

trainer = L.Trainer(max_epochs=15, accelerator="gpu", devices=4, strategy='ddp_notebook', num_nodes=1)

trainer.fit(model, train_loader, validation_loader)

# %%
# Calculate the train and validation loss
train_loss = trainer.validate(model, train_loader)
validation_loss = trainer.validate(model, validation_loader)

# %%
# Calculate the test loss
test_loss = trainer.validate(model, test_loader)


