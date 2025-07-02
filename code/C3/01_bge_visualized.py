import torch
from visual_bge.modeling import Visualized_BGE
import numpy as np

model = Visualized_BGE(model_name_bge = "BAAI/bge-base-en-v1.5", model_weight="../../models/bge/Visualized_base_en_v1.5.pth")
model.eval()
with torch.no_grad():
    query_emb = model.encode(image="../../data/C3/imgs/cir_query.png", text="Make the background dark, as if the camera has taken the photo at night")
    candi_emb_1 = model.encode(image="../../data/C3/imgs/cir_candi_1.png")
    candi_emb_2 = model.encode(image="../../data/C3/imgs/cir_candi_2.png")

sim_1 = query_emb @ candi_emb_1.T
sim_2 = query_emb @ candi_emb_2.T
print(sim_1, sim_2) # tensor([[0.8750]]) tensor([[0.7816]])
