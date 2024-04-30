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
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.plugins.environments import SLURMEnvironment  # type: ignore
from models import MLPModel
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


def create_datasets(train_dataset, validation_dataset, test_dataset):
    train_dataset = AntibodyDataset(train_dataset)
    validation_dataset = AntibodyDataset(validation_dataset)
    test_dataset = AntibodyDataset(test_dataset)

    return train_dataset, validation_dataset, test_dataset


def create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    time.sleep(args.job_array_num * 5)
    seed_everything(7850)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

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
        ["Sequence", "antigen"],
    ]
    esm_models = ["8M", "35M", "150M", "650M", "3B"]

    column_combination = column_combinations[args.job_array_num - 1]

    for esm_model in esm_models:
        print("----------")
        print(f"Current column combination: {column_combination}")
        print(f"Current esm_model: {esm_model}")
        print("----------")
        # Create a dataset class for the train, validation, and test datasets
        train_dataset, validation_dataset, test_dataset = create_train_test_val(
            [f"{column}_embeddings" for column in column_combination], esm_model
        )
        train_dataset, validation_dataset, test_dataset = create_datasets(
            train_dataset, validation_dataset, test_dataset
        )

        # Create dataloaders
        train_loader, validation_loader, test_loader = create_dataloaders(
            train_dataset, validation_dataset, test_dataset, batch_size=32
        )

        # Train model using PyTorch Lightning
        model = MLPModel(
            input_dim=train_dataset[0][0].shape[0],
            dropout=0.1,
            esm_model=esm_model,
            columns=",".join(column_combination),
        )
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        trainer = L.Trainer(
            max_epochs=15,
            strategy="ddp",
            accelerator="gpu",
            num_nodes=1,
            devices=4,
            plugins=[SLURMEnvironment(auto_requeue=False)],
            enable_progress_bar=True,
            callbacks=[es],
        )

        trainer.fit(model, train_loader, validation_loader)

        # Calculate the train and validation loss
        train_loss = trainer.validate(model, train_loader)
        validation_loss = trainer.validate(model, validation_loader)

        # Calculate the test loss
        test_loss = trainer.validate(model, test_loader)

        # Write the results to a file
        results = {
            "esm_model": esm_model,
            "columns": column_combination,
            "train_results": train_loss[0],
            "validation_results": validation_loss[0],
            "test_results": test_loss[0],
        }

        # Make results folder
        os.makedirs("mlp_results", exist_ok=True)

        # Make esm model folder inside results folder
        os.makedirs(f"mlp_results/{esm_model}", exist_ok=True)

        # Write results to file
        with open(
            f'mlp_results/{esm_model}/{",".join(column_combination)}.json', "w"
        ) as f:
            json.dump(results, f)

        # Delete all variables used
        del train_dataset
        del validation_dataset
        del test_dataset
        del train_loader
        del validation_loader
        del test_loader
        del model
        del es
        del trainer
        del train_loss
        del validation_loss
        del test_loss
        del results
