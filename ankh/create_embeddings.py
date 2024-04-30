# %%
import csv
import os

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import ankh

# %%
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Distribute model across multiple GPUs if available
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")


# %%
def embed_and_save(gpu_id, seq, col):
    local_device = torch.device(f"cuda:{gpu_id}")

    # Load model and tokenizer on the specified GPU
    model, tokenizer = ankh.load_base_model()
    model.to(local_device)
    model.eval()

    # Embedding operation
    outputs = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        embeddings = model(
            input_ids=outputs["input_ids"].to(local_device),
            attention_mask=outputs["attention_mask"].to(local_device),
        )

    embeddings = np.array(embeddings)

    # Save the embedding
    sequence = seq[0]
    np.save(f"./embeddings/{col}/{sequence}.npy", embeddings)


# %%
if not os.path.exists("./embeddings"):
    os.makedirs("./embeddings")

# Loop through each category and process sequences in parallel
for col in [
    "Sequence",
    "HC",
    "LC",
    "CDRH1",
    "CDRH2",
    "CDRH3",
    "CDRL1",
    "CDRL2",
    "CDRL3",
]:
    tasks = []

    # Load all sequences for the current column
    with open(f"./sequences/{col}.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        sequences = [row[0] for row in reader]

        print(f"Loaded {len(sequences)} sequences from {col}")

        if not os.path.exists(f"./embeddings/{col}"):
            os.makedirs(f"./embeddings/{col}")

        # Assign GPU and build tasks for this column
        for idx, sequence in enumerate(sequences):
            if not os.path.exists(f"./embeddings/{col}/{sequence}.npy"):
                gpu_id = idx % num_gpus  # Round-robin GPU assignment
                tasks.append((gpu_id, [sequence], col))

    # Use thread_map to process tasks in parallel across GPUs
    if tasks:
        thread_map(lambda x: embed_and_save(x[0], x[1], x[2]), tasks)

print("Parallelized embedding process completed.")

# %%
# embed_and_save(0, ["GFTLNSYGIS"], "Sequence")
