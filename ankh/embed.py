import csv
import hashlib  # For generating hash
import os

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

import ankh

# Determine the device to use for computations
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device: {}".format(device))

# Load the model and tokenizer
model, tokenizer = ankh.load_base_model()
model.to(device)
model.eval()


# Function to embed sequences and save them to disk
def embed_and_save(seq, col):
    outputs = tokenizer.batch_encode_plus(
        seq,
        add_special_tokens=True,
        padding=True,
        is_split_into_words=False,
        return_tensors="pt",
    )

    with torch.no_grad():
        embeddings = model(
            input_ids=outputs["input_ids"].to(device),
            attention_mask=outputs["attention_mask"].to(device),
        )

    # Move embeddings to CPU
    embeddings = np.array(embeddings)

    # Generate a hash-based filename to prevent file name too long errors
    sequence = seq[0]
    sequence_hash = hashlib.md5(
        sequence.encode()
    ).hexdigest()  # MD5 for generating a unique hash
    filepath = f"./embeddings/{col}/{sequence_hash}.npz"

    try:
        # Save the embedding
        np.savez_compressed(filepath, embeddings)
    except OSError as e:
        if e.errno == 36:  # File name too long
            print(
                f"Error: File name too long. File path: {filepath}. Trying with a hash."
            )
            np.savez_compressed(
                filepath, embeddings
            )  # Attempt to save with a shorter name
        else:
            raise  # re raise exception
    except Exception as e:
        print("Unknown error", e)
        print("Unable to save", seq)
        return


# Create the embeddings folder if it doesn't exist
if not os.path.exists("./embeddings"):
    os.makedirs("./embeddings")

# Process each column
columns = ["Sequence", "HC", "LC", "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]
for col in columns:
    tasks = []

    # Load all the sequences from CSV
    with open(f"./sequences/{col}.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        sequences = [row[0] for row in reader]

        print(f"Loaded {len(sequences)} sequences from {col}")

        # Create a subfolder for the embeddings
        if not os.path.exists(f"./embeddings/{col}"):
            os.makedirs(f"./embeddings/{col}")

        # Add sequences to tasks
        for sequence in sequences:
            sequence_hash = hashlib.md5(
                sequence.encode()
            ).hexdigest()  # Generate a unique hash
            filepath = f"./embeddings/{col}/{sequence_hash}.npz"
            if not os.path.exists(filepath):  # If it doesn't already exist
                tasks.append(([sequence], col))

    # Embed and save using threading
    thread_map(lambda x: embed_and_save(x[0], x[1]), tasks)
