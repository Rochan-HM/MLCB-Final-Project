import os

import pandas as pd
import torch
from tqdm import tqdm

import esm


def calculate_embeddings(
    df, batch_size, model, alphabet, batch_converter, device, layer, pbar_name
):
    # Define embeddings and batch size
    sequence_representations = []

    model.eval()
    model = model.to(device)

    # Hashmap to check if sequence already has an embedding
    seq_to_embedding = {}

    # Calculate embeddings
    try:
        for i in tqdm(range(0, len(df), batch_size), desc=pbar_name):
            data = df.iloc[i : i + batch_size]
            # print(data)
            # print(data['seq'])
            # print(data['seq'].iloc[0])
            sequence = data["seq"].iloc[0]
            if sequence in seq_to_embedding:
                sequence_representations.append(seq_to_embedding[sequence])
                continue
            _, _, batch_tokens = batch_converter(data.values.tolist())
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[layer], return_contacts=True)
            token_representations = results["representations"][layer].cpu()
            for i, tokens_len in enumerate(batch_lens):
                embedding = token_representations[i, 1 : tokens_len - 1].mean(0)
                seq_to_embedding[sequence] = embedding
                sequence_representations.append(embedding)
    except Exception as e:
        # print(f"Received error message {e}")
        raise e
        return None
    return sequence_representations


if __name__ == "__main__":
    # Load dataset
    path = "/home/hice1/gdeshpande9/scratch/Project/Dataset/antibody_dataset_1/MITLL_AAlphaBio_Ab_Binding_dataset.csv"
    dataset = pd.read_csv(path)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # functions = ['esm2_t6_8M_UR50D', 'esm2_t12_35M_UR50D', 'esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']
    # layers = [5, 11, 29, 32, 35, 47]

    functions = ["esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"]
    layers = [35, 47]

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
    ]

    for layer, function in zip(layers, functions):
        print(f"Loading in model {function}")
        model, alphabet = getattr(esm.pretrained, function)()

        for column in sequence_columns:
            # print(dataset[column])
            df_column = dataset[column].reset_index(name="seq")
            # print(df_column)
            batch_converter = alphabet.get_batch_converter(truncation_seq_length=300)

            pbar_name = f"{function}-{column}"
            column_embeddings = calculate_embeddings(
                df_column, 1, model, alphabet, batch_converter, device, layer, pbar_name
            )

            dataset[f"{column}_embeddings"] = column_embeddings

        # Save dataset as compressed pickle
        os.makedirs(f"./data/{function}", exist_ok=True)

        dataset.to_pickle(f"./data/{function}/dataset.pkl")
