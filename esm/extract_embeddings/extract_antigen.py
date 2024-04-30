# %%
import numpy as np
import protenc

# %%
print(protenc.list_models())

# %%
model = "esm2_t36_3B_UR50D"
encoder = protenc.get_encoder(model, device="cuda")
protein = ["PDVDLGDISGINAS"]

for embed in encoder(protein, return_format="numpy"):
    e = embed.mean(0)
    print(e)
    with open(f"./data/{model.replace('_t36', '')}-antigen.npy", "wb") as f:
        np.save(f, e)

# %%
