# Code for MLCB final project

## Group 21
- **Group members**: 
    - **Rochan Madhusudhana**
    - **Gururaj Deshpande**

## Project Installation

Please install [Anaconda](https://www.anaconda.com/products/individual) before running the code.

Dependencies can be installed with
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia; conda install -y pandas numpy; pip install fair-esm; conda install -y plotly matplotlib seaborn; conda install -y tqdm ipykernel jupyter ipywidgets
```

## Project Structure

The project is structured as follows:

```
.
├── ankh (Ankh related code)
│   ├── CreateAnkhEmbeddings.sbatch
│   ├── create_embeddings.py
│   ├── embed.py
│   ├── fine_tune.py
│   └── regressor.py
├── esm (ESM related code)
│   ├── Dataset
│   ├── exploration.py
│   ├── extract_embeddings
│   │   ├── data
│   │   ├── extract_antigen.py
│   │   ├── extract_embeddings.py
│   │   └── start_embedding_extraction.sbatch
│   └── train_models
│       ├── add_cols.py
│       ├── dataset.py
│       ├── logs
│       ├── mlp_results
│       ├── models.py
│       ├── start_mlp_model_training.sbatch
│       ├── start_model_training.sbatch
│       ├── start_rf_model_training.sbatch
│       ├── train_mlp_models.py
│       ├── train_models.py
│       └── train_rf_models.py
└── readme.md
```

## Project Running

Please run the corresponding python file using the following command:
```bash
python <filename>.py
```

Any `.sbatch` file can be run using the following command:
```bash
sbatch <filename>.sbatch
```
