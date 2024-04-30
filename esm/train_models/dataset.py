import pandas as pd
import torch


class AntibodyDataset(torch.utils.data.Dataset):
    def __init__(self, data, concat_embeddings_into_1=True):
        self.concat_embeddings = concat_embeddings_into_1
        data = self.process(data)
        self.data = data

    def process(self, data):
        embeddings_columns = [col for col in data.columns if "embedding" in col]

        if self.concat_embeddings:
            # Iterate through the dataframe and get the Tensor stored in the sequence_embedding column
            all_embeddings = []
            for col in embeddings_columns:
                col_embeddings = []
                for i in range(len(data)):
                    item = data.iloc[i][col]
                    if not isinstance(item, torch.Tensor):
                        item = torch.tensor(item, dtype=torch.float32)
                    col_embeddings.append(item)
                col_embeddings = torch.stack(col_embeddings)
                all_embeddings.append(col_embeddings)

            # Concatenate all the embeddings in all_embeddings horizontally
            sequence_embeddings = torch.cat(all_embeddings, dim=1)

        else:
            all_embeddings = []

            # Iterate through all the rows in the dataframe and vertically stack the embeddings
            for i in range(len(data)):
                row_embeddings = []
                for col in embeddings_columns:
                    item = data.iloc[i][col]
                    if not isinstance(item, torch.Tensor):
                        item = torch.tensor(item, dtype=torch.float32)
                    row_embeddings.append(item)
                row_embeddings = torch.stack(row_embeddings)
                all_embeddings.append(row_embeddings)

            # Stack the embeddings in the first dimension
            sequence_embeddings = torch.stack(all_embeddings)

        # Create a tensor of the target
        pred_affinity = torch.tensor(data["Pred_affinity"].values, dtype=torch.float32)

        return torch.utils.data.TensorDataset(sequence_embeddings, pred_affinity)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sequence_embedding and label at the given index
        sequence_embedding, target = self.data[idx]
        target = target.unsqueeze(0)
        return sequence_embedding, target
