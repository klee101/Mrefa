import torch

from flame_pytorch import FLAME, get_config

config = get_config()  
flame_model = FLAME(config)

shape_params = torch.zeros(8, config.shape_params, dtype=torch.float32)
expression_params = torch.zeros(8, config.expression_params, dtype=torch.float32)
pose_params = torch.zeros(8, config.pose_params, dtype=torch.float32)

vertices, landmarks = flame_model(shape_params, expression_params, pose_params)

scripted_model.save("flame_model.pt")
