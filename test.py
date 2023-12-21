from compressai.models.MCM import MCM

import torch 

model = MCM()
imgs = torch.rand((1, 3, 224, 224))
total_score = torch.rand((1, 196))

out = model.forward(imgs, total_score)
