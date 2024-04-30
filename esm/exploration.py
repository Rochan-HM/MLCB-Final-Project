# %% [markdown]
# # Exploratory Data Analysis of the Antibody Dataset

# %%
# Imports
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm.notebook import tqdm

# %% [markdown]
# ## Load in Dataset

# %%
dataset1 = pd.read_csv(
    "./Dataset/antibody_dataset_1/MITLL_AAlphaBio_Ab_Binding_dataset.csv"
)
dataset2 = pd.read_csv(
    "./Dataset/antibody_dataset_2/MITLL_AAlphaBio_Ab_Binding_dataset2.csv"
)

# %%
# View dataset1
dataset1

# %%
# View dataset2
dataset2

# %%
dataset1.describe()

# %%
dataset2.describe()

# %%
# Remove NaN Pred_affinity rows
dataset1 = dataset1[dataset1["Pred_affinity"].notna()]
dataset1

# %%
# Remove NaN Pred_affinity rows
dataset2 = dataset2[dataset2["Pred_affinity"].notna()]
dataset2

# %%
# Analyze number of unique rows with only sequence columns for dataset1
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
sequence_dataset = dataset1[sequence_columns]
original_length = len(sequence_dataset)
sequence_dataset = sequence_dataset.drop_duplicates()
unique_length = len(sequence_dataset)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %%
# Analyze number of unique rows with only sequence columns for dataset2
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
sequence_dataset = dataset2[sequence_columns]
original_length = len(sequence_dataset)
sequence_dataset = sequence_dataset.drop_duplicates()
unique_length = len(sequence_dataset)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %% [markdown]
# ## Understanding relationship between POI and sequence columns in both datasets

# %%
# Create a new column with combined sequences for dataset1
dataset1_with_combined = dataset1.copy()
dataset1_with_combined["Combined"] = (
    dataset1["HC"]
    + dataset1["CDRH1"]
    + dataset1["CDRH2"]
    + dataset1["CDRH3"]
    + dataset1["LC"]
    + dataset1["CDRL1"]
    + dataset1["CDRL2"]
    + dataset1["CDRL3"]
)

# Create a new column with combined sequences for dataset2
dataset2_with_combined = dataset2.copy()
dataset2_with_combined["Combined"] = (
    dataset2["HC"]
    + dataset2["CDRH1"]
    + dataset2["CDRH2"]
    + dataset2["CDRH3"]
    + dataset2["LC"]
    + dataset2["CDRL1"]
    + dataset2["CDRL2"]
    + dataset2["CDRL3"]
)


# %%
# Remove all columns except Combined and POI for dataset1
dataset1_with_combined_poi = dataset1_with_combined[["Combined", "POI"]]

# Calculate unique rows for dataset1
original_length = len(dataset1_with_combined_poi)
dataset1_with_combined_poi = dataset1_with_combined_poi.drop_duplicates()
unique_length = len(dataset1_with_combined_poi)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %%
# Remove all columns except Combined and POI for dataset2
dataset2_with_combined_poi = dataset2_with_combined[["Combined", "POI"]]

# Calculate unique rows for dataset2
original_length = len(dataset2_with_combined_poi)
dataset2_with_combined_poi = dataset2_with_combined_poi.drop_duplicates()
unique_length = len(dataset2_with_combined_poi)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %% [markdown]
# ## Understanding relationship between Target and sequence columns in both datasets

# %%
# Remove all columns except Combined and Target for dataset1
dataset1_with_combined_target = dataset1_with_combined[["Combined", "Target"]]

# Calculate unique rows for dataset1
original_length = len(dataset1_with_combined_target)
dataset1_with_combined_target = dataset1_with_combined_target.drop_duplicates()
unique_length = len(dataset1_with_combined_target)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %%
# Remove all columns except Combined and Target for dataset2
dataset2_with_combined_target = dataset2_with_combined[["Combined", "Target"]]

# Calculate unique rows for dataset2
original_length = len(dataset2_with_combined_target)
dataset2_with_combined_target = dataset2_with_combined_target.drop_duplicates()
unique_length = len(dataset2_with_combined_target)
print(f"Original Length: {original_length}")

# %% [markdown]
# ## Combine Datasets 1 and 2

# %%
dataset = dataset1

# %% [markdown]
# ## Analyze unique rows for sequence columns in combined dataset

# %%
# Analyze number of unique rows with only sequence columns
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
sequence_dataset = dataset[sequence_columns]
original_length = len(sequence_dataset)
sequence_dataset = sequence_dataset.drop_duplicates()
unique_length = len(sequence_dataset)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %%
# Add pred_affinity to sequence dataset
sequence_dataset = dataset[sequence_columns]
sequence_dataset["Pred_affinity"] = dataset["Pred_affinity"]
original_length = len(sequence_dataset)
sequence_dataset = sequence_dataset.drop_duplicates()
unique_length = len(sequence_dataset)
print(f"Original Length: {original_length}")
print(f"Unique Length: {unique_length}")

# %% [markdown]
# ## Analyze Duplicate Sequence Rows and Pred_affinity distribution

# %%
pred_affinity_distribution = []

# Create the combined column in the sequence_dataset
sequence_dataset["Combined"] = sequence_dataset[sequence_columns].apply(
    lambda row: "".join(row.values.astype(str)), axis=1
)
unique_sequences = sequence_dataset["Combined"].unique()

# Get all the counts, means, and standard deviations of pred_affinity for each unique combined sequence using groupby
sequence_to_pred_affinity_count = (
    sequence_dataset.drop(columns=sequence_columns).groupby("Combined").count()
)
sequence_to_pred_affinity_mean = (
    sequence_dataset.drop(columns=sequence_columns).groupby("Combined").mean()
)
sequence_to_pred_affinity_std = (
    sequence_dataset.drop(columns=sequence_columns).groupby("Combined").std()
)

# Combine the counts, means, and standard deviations into a single dataframe
sequence_to_pred_affinity = pd.concat(
    [
        sequence_to_pred_affinity_count,
        sequence_to_pred_affinity_mean,
        sequence_to_pred_affinity_std,
    ],
    axis=1,
)
sequence_to_pred_affinity.columns = ["Count", "Mean", "Std"]

# %%
sequence_to_pred_affinity

# %%
# Create a Plotly Violin Plot to show the distribution of the number of pred_affinities from each unique sequence
unique_counts = sequence_to_pred_affinity["Count"].values

fig = go.Figure(
    data=go.Violin(
        y=unique_counts,
        box_visible=True,
        meanline_visible=True,
        x0="Unique Sequences",
        name="Unique Sequences",
    )
)
fig.update_layout(
    title="Distribution of Number of Pred_affinities from each Unique Sequence",
    yaxis_title="Number of Pred_affinities",
    xaxis_title="Unique Sequences",
)
fig.show()

# %%
# Create a Plotly Violin Plot to show the distribution of the mean pred_affinities from each unique sequence
unique_means = sequence_to_pred_affinity["Mean"].values

fig = go.Figure(
    data=go.Violin(
        y=unique_means,
        box_visible=True,
        meanline_visible=True,
        x0="Unique Sequences",
        name="Unique Sequences",
    )
)
fig.update_layout(
    title="Distribution of Mean Pred_affinities from each Unique Sequence",
    yaxis_title="Mean Pred_affinity",
    xaxis_title="Unique Sequences",
)
fig.show()

# %%
# Create a Plotly Violin Plot to show the distribution of the standard deviation of pred_affinities from each unique sequence
unique_stds = sequence_to_pred_affinity["Std"].values

fig = go.Figure(
    data=go.Violin(
        y=unique_stds,
        box_visible=True,
        meanline_visible=True,
        x0="Unique Sequences",
        name="Unique Sequences",
    )
)
fig.update_layout(
    title="Distribution of Standard Deviation of Pred_affinities from each Unique Sequence",
    yaxis_title="Standard Deviation of Pred_affinity",
    xaxis_title="Unique Sequences",
)
fig.show()

# %% [markdown]
# ## Analyze Targets

# %%
# Get overall
targets = dataset["Target"]
targets.describe()

# %%
# Get unique targets
targets.unique()

# %% [markdown]
# ## Analyze Amino Acid Sequences


# %%
def analyze_sequence_column(column_name: str, dataset: pd.DataFrame):
    # Get sequence
    sequences = dataset[column_name]
    description = sequences.describe()

    # Get length of sequences
    sequence_lengths = sequences.apply(len)

    # Create violin plot
    fig = go.Figure(
        data=go.Violin(
            y=sequence_lengths,
            box_visible=True,
            meanline_visible=True,
            x0="Length",
            name="Length",
        )
    )
    fig.update_layout(
        title=f"{column_name} Distribution",
    )

    return fig, description


# %% [markdown]
# ### Sequence Column

# %%
# Describe sequence
fig, description = analyze_sequence_column("Sequence", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### HC

# %%
# Describe HC
fig, description = analyze_sequence_column("HC", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### LC

# %%
# Describe LC
fig, description = analyze_sequence_column("LC", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRH1

# %%
# Describe CDRH1
fig, description = analyze_sequence_column("CDRH1", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRH2

# %%
# Describe CDRH2
fig, description = analyze_sequence_column("CDRH2", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRH3

# %%
# Describe CDRH3
fig, description = analyze_sequence_column("CDRH3", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRL1

# %%
# Describe CDRL1
fig, description = analyze_sequence_column("CDRL1", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRL2

# %%
# Describe CDRL2
fig, description = analyze_sequence_column("CDRL2", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ### CDRL3

# %%
# Describe CDRL3
fig, description = analyze_sequence_column("CDRL3", dataset)

# Print description
print(description)

# Show plot
fig.show()

# %% [markdown]
# ## Analyze Predicted Affinity

# %%
# Get all predicted affinity values
pred_affinities = dataset["Pred_affinity"]
description = pred_affinities.describe()

# %%
# Create a violin plot
fig = go.Figure(
    data=go.Violin(
        y=pred_affinities,
        box_visible=True,
        meanline_visible=True,
        x0="Pred_affinity",
        name="Pred_affinity",
    )
)
fig.update_layout(
    title="Pred_affinity Distribution",
)
fig.show()
