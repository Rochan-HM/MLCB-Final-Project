import lightning as L
import torch
import torch.nn as nn
from torchmetrics.regression import SpearmanCorrCoef


class MLPModel(L.LightningModule):
    def __init__(self, input_dim, dropout=0.1, esm_model=None, columns=None):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )
        self.spearman_correlation = SpearmanCorrCoef()

        # Save esm2-model and length as hparams
        hyperparams = {"esm2_model": esm_model, "columns": columns}
        self.save_hyperparameters(hyperparams)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.model(x), y)

        corr = self.spearman_correlation(self.model(x), y)
        self.log("train_loss", loss)
        self.log("train_corr", corr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = torch.nn.functional.mse_loss(self.model(x), y)
        corr = self.spearman_correlation(self.model(x), y)
        self.log("val_loss", loss)
        self.log("val_corr", corr, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class MLPModelWithAttention(L.LightningModule):
    def __init__(
        self, num_sequences, embedding_dim, dropout=0.1, esm_model=None, columns=None
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_sequences = num_sequences

        # Attention Components
        self.query_projection = nn.Linear(self.embedding_dim, self.embedding_dim // 2)
        self.key_projection = nn.Linear(self.embedding_dim, self.embedding_dim // 2)

        # Main Model Layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1),
        )

        self.spearman_correlation = SpearmanCorrCoef()

        # Save esm2-model and length as hparams
        hyperparams = {
            "esm2_model": esm_model,
            "columns": columns,
        }
        self.save_hyperparameters(hyperparams)

    def forward(self, x):
        # Calculate attention weights
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.embedding_dim // 2)
        )
        attention_weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention
        weighted_embeddings = torch.matmul(attention_weights, x)

        # Aggregate across sequences for each sample in the batch
        batch_predictions = torch.sum(
            weighted_embeddings, dim=1
        )  # Sum across the sequence dimension

        # Pass individual predictions through the output layers
        output = self.output_layer(batch_predictions)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        result = self.forward(x)
        loss = torch.nn.functional.mse_loss(self.forward(x), y)
        corr = self.spearman_correlation(self.forward(x), y)
        self.log("train_loss", loss)
        self.log("train_corr", corr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        result = self.forward(x)
        loss = torch.nn.functional.mse_loss(self.forward(x), y)
        corr = self.spearman_correlation(self.forward(x), y)
        self.log("val_loss", loss)
        self.log("val_corr", corr, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
