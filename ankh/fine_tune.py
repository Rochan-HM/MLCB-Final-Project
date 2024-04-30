# %%
import concurrent.futures
import hashlib  # For generating hash-based filenames
import os
import random
from functools import partial

import numpy as np
import pandas as pd
import torch
import transformers.models.convbert as c_bert
from datasets import load_dataset
from scipy import stats

# Train test split
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map
from transformers import EvalPrediction, Trainer, TrainingArguments

import ankh

# %%
# Seed for reproducibility
seed = 7850
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Available device:", device)


# %%
def load_embedding(col, sequence):
    sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
    filepath = f"./embeddings/{col}/{sequence_hash}.npz"
    with np.load(filepath, allow_pickle=True) as data:
        embedding = data["arr_0"]
    # embedding = np.load(filepath, allow_pickle=True)
    return embedding


# %%
df1 = pd.read_csv("data/data.csv")

# Do a 70, 15, 15 split
train, test = train_test_split(df1, test_size=0.3, random_state=seed)
val, test = train_test_split(test, test_size=0.5, random_state=seed)

train_sequences = train["Sequence"].tolist()
train_labels = train["Pred_affinity"].tolist()

val_sequences = val["Sequence"].tolist()
val_labels = val["Pred_affinity"].tolist()

test_sequences = test["Sequence"].tolist()
test_labels = test["Pred_affinity"].tolist()

# %%
# load_embedding("Sequence", train_sequences[1])

# %%
# train_embeddings = load_embeddings_parallel("Sequence", train_sequences)
# val_embeddings = load_embeddings_parallel("Sequence", val_sequences)
# test_embeddings = load_embeddings_parallel("Sequence", test_sequences)

# Prepare arguments for thread map
train_tasks = [seq for seq in train_sequences]
val_tasks = [seq for seq in val_sequences]
test_tasks = [seq for seq in test_sequences]

load_embedding_sequence_col = partial(load_embedding, "Sequence")

# Parallelize loading embeddings
train_embeddings = thread_map(load_embedding_sequence_col, train_tasks, max_workers=2)
val_embeddings = thread_map(load_embedding_sequence_col, val_tasks, max_workers=2)
test_embeddings = thread_map(load_embedding_sequence_col, test_tasks, max_workers=2)


# %%
class EmbeddingsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __getitem__(self, idx):
        embedding = self.sequences[idx]
        label = self.labels[idx]
        return {
            "embed": torch.tensor(embedding),
            "labels": torch.tensor(label, dtype=torch.float32).unsqueeze(-1),
        }

    def __len__(self):
        return len(self.labels)


# %%
training_dataset = EmbeddingsDataset(train_embeddings, train_labels)
validation_dataset = EmbeddingsDataset(val_embeddings, val_labels)
testing_dataset = EmbeddingsDataset(test_embeddings, test_labels)


# %%
def model_init(embed_dim, training_labels_mean=None):
    hidden_dim = int(embed_dim / 2)
    num_hidden_layers = 3
    num_layers = 3
    nhead = 4
    dropout = 0.2
    conv_kernel_size = 7
    pooling = "max"  # Possible options are 'avg' and 'max'

    model = ankh.ConvBertForRegression(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_hidden_layers=num_hidden_layers,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        training_labels_mean=training_labels_mean,
    )
    return model.to(device)


# %%
def compute_metrics(p: EvalPrediction):
    return {
        "spearmanr": stats.spearmanr(p.label_ids, p.predictions).correlation,
    }


# %%
training_args = TrainingArguments(
    output_dir="./ankh_fine_tune",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=1000,
    learning_rate=1e-03,
    weight_decay=0.0,
    logging_dir="./ankh_fine_tune/logs",
    logging_steps=200,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=8,
    fp16=False,
    seed=seed,
    load_best_model_at_end=True,
    metric_for_best_model="eval_spearmanr",
)

# %%
trainer = Trainer(
    model_init=partial(model_init, embed_dim=train_embeddings[0].shape[1]),
    args=training_args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()
