import numpy as np
import pyrender
import torch
import trimesh

from flame_pytorch import FLAME, get_config

# Load FLAME configuration and initialize model
config = get_config()
flame_model = FLAME(config).cuda()

# Create default shape, expression, and pose parameters
shape_params = torch.zeros(8, config.shape_params, dtype=torch.float32).cuda()
expression_params = torch.zeros(8, config.expression_params, dtype=torch.float32).cuda()
pose_params = torch.zeros(8, config.pose_params, dtype=torch.float32).cuda()

# Forward pass to get vertices
vertices, _ = flame_model(shape_params, expression_params, pose_params)
vertices_np = vertices[0].detach().cpu().numpy()

# Get FLAME faces
faces = flame_model.faces

# Create and visualize the mesh
vertex_colors = np.ones([vertices_np.shape[0], 4]) * [0.7, 0.7, 0.7, 1.0]
tri_mesh = trimesh.Trimesh(vertices_np, faces, vertex_colors=vertex_colors)
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)