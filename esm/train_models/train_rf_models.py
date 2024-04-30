# Imports
import argparse
import json
import os
import time

import lightning as L
import numpy as np
import pandas as pd
import requests
import torch
from dataset import AntibodyDataset
from lightning.pytorch import seed_everything
from models import MLPModel, MLPModelWithAttention
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm.notebook import tqdm

import esm

parser = argparse.ArgumentParser(description="Schedule ablation study for MLP Model")
parser.add_argument("--job_array_num", type=int, required=True, help="Job Array Num")
args = parser.parse_args()


def create_train_test_val(columns_to_include, esm_model):
    # Load in ESM 8M dataset
    path = f"../extract_embeddings/data/esm2_{esm_model}_UR50D/dataset2.pkl"
    dataset = pd.read_pickle(path)

    # Remove non-target rows
    dataset = dataset[dataset["Target"] == "MIT_Target"]

    # Remove NaN Pred_affinity rows
    dataset = dataset[dataset["Pred_affinity"].notna()]

    # Remove all sequence columns but keep embeddings columns
    dataset = dataset.drop(
        columns=[
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
    )

    # Remove all columns that don't have embeddings in name and are not Pred_affinity
    dataset = dataset.loc[:, dataset.columns.str.contains("embeddings|Pred_affinity")]

    # Remove all embeddings except ones selected
    dataset = dataset[columns_to_include + ["Pred_affinity"]]

    # Randomly sample 50% of the dataset
    dataset = dataset.sample(frac=0.5)

    # Create train, validation, and test datasets using a 70/15/15 split
    train_dataset = dataset.sample(frac=0.7)
    dataset = dataset.drop(train_dataset.index)
    validation_dataset = dataset.sample(frac=0.5)
    test_dataset = dataset.drop(validation_dataset.index)

    return train_dataset, validation_dataset, test_dataset


def combine_embeddings(data):
    embeddings_columns = [col for col in data.columns if "embedding" in col]
    all_embeddings = []
    for col in embeddings_columns:
        col_embeddings = []
        for i in range(len(data)):
            col_embeddings.append(data.iloc[i][col])
        col_embeddings = torch.stack(col_embeddings)
        all_embeddings.append(col_embeddings)
    sequence_embeddings = torch.cat(all_embeddings, dim=1)
    pred_affinity = torch.tensor(data["Pred_affinity"].values, dtype=torch.float32)
    return (
        sequence_embeddings.detach().cpu().numpy(),
        pred_affinity.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    time.sleep(args.job_array_num * 5)
    seed_everything(7850)

    # Define sequence_columns
    sequence_columns = [
        "Sequence",
        "HC",
        "LC",
        "CDRH1",
        "CDRH2",
        "CDRH3",
        "CDRL1",
        "CDRL2",
        "CDRL3",
        "antigen",
    ]

    # Define column_combinations
    column_combinations = [
        [
            "Sequence",
            "HC",
            "LC",
            "CDRH1",
            "CDRH2",
            "CDRH3",
            "CDRL1",
            "CDRL2",
            "CDRL3",
            "antigen",
        ],
        ["HC", "antigen"],
        ["LC", "antigen"],
        ["CDRH1", "antigen"],
        ["CDRH2", "antigen"],
        ["CDRH3", "antigen"],
        ["CDRL1", "antigen"],
        ["CDRL2", "antigen"],
        ["CDRL3", "antigen"],
        ["Sequence", "HC", "LC", "antigen"],
        ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3", "antigen"],
        ["CDRH1", "CDRH2", "CDRH3", "antigen"],
        ["CDRL1", "CDRL2", "CDRL3", "antigen"],
    ]
    esm_models = ["8M", "35M", "150M", "650M", "3B"]

    column_combination = column_combinations[args.job_array_num - 1]
    column_combination = [f"{column}_embeddings" for column in column_combination]

    for esm_model in esm_models:
        print("----------")
        print(f"Current column combination: {column_combination}")
        print(f"Current esm_model: {esm_model}")
        print("----------")
        # Create a dataset class for the train, validation, and test datasets
        train_dataset, validation_dataset, test_dataset = create_train_test_val(
            column_combination, esm_model
        )
        X_train, y_train = combine_embeddings(train_dataset)
        X_val, y_val = combine_embeddings(validation_dataset)
        X_test, y_test = combine_embeddings(test_dataset)

        # Train model using sklearn
        regr = RandomForestRegressor(
            n_jobs=-1,
            random_state=7850,
            n_estimators=100,
            max_depth=7,
            criterion="friedman_mse",
        )
        regr.fit(X_train, y_train)

        train_predictions = regr.predict(X_train)
        val_predictions = regr.predict(X_val)
        test_predictions = regr.predict(X_test)

        train_corr = spearmanr(train_predictions, y_train)
        train_loss = mean_squared_error(train_predictions, y_train)
        val_corr = spearmanr(val_predictions, y_val)
        val_loss = mean_squared_error(val_predictions, y_val)
        test_corr = spearmanr(test_predictions, y_test)
        test_loss = mean_squared_error(test_predictions, y_test)

        results = {
            "train": {
                "loss": train_loss,
                "corr": train_corr.correlation,
            },
            "validation": {
                "loss": val_loss,
                "corr": val_corr.correlation,
            },
            "test": {
                "loss": test_loss,
                "corr": test_corr.correlation,
            },
        }

        # Make results folder
        os.makedirs("rf_results", exist_ok=True)

        # Make esm model folder inside results folder
        os.makedirs(f"rf_results/{esm_model}", exist_ok=True)

        # Write results to file
        with open(
            f'rf_results/{esm_model}/{",".join(column_combination)}.json', "w"
        ) as f:
            json.dump(results, f)
