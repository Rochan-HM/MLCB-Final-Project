# Imports
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

models = [
    "esm2_8M_UR50D",
    "esm2_35M_UR50D",
    "esm2_150M_UR50D",
    "esm2_650M_UR50D",
    "esm2_3B_UR50D",
]

for model in tqdm(models):
    print(f"Loading model: {model}")
    base_path = f"../extract_embeddings/data/{model}"
    dataset = pd.read_pickle(base_path + "/dataset.pkl")

    print("Loaded dataset")
    antigen_path = base_path + "-antigen.npy"
    antigen_embeddings = np.load(antigen_path, allow_pickle=True)
    print("Loaded antigen embed")

    num_rows = dataset.shape[0]

    print("converting to torch")
    antigen_tensor = torch.tensor(antigen_embeddings)
    antigen_embeddings_tensor = antigen_tensor.expand(num_rows, -1)

    print("Adding to df")
    dataset["antigen_embeddings"] = [
        antigen_embeddings_tensor[i] for i in range(num_rows)
    ]
    print("Added to df")

    dataset.to_pickle(base_path + "/dataset2.pkl")
    print(f"Done model: {model}")
