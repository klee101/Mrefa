import os
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from argparse import Namespace

import numpy as np
import torch
import trimesh
from flame_pytorch import FLAME, get_config

app = FastAPI()

def get_default_config():
    return Namespace(
        flame_model_path="./model/generic_model.pkl",
        static_landmark_embedding_path="./model/flame_static_embedding.pkl",
        dynamic_landmark_embedding_path="./model/flame_dynamic_embedding.npy",
        shape_params=100,
        expression_params=50,
        pose_params=6,
        use_face_contour=True,
        use_3D_translation=True,
        optimize_eyeballpose=True,
        optimize_neckpose=True,
        num_worker=4,
        batch_size=8,
        ring_margin=0.5,
        ring_loss_weight=1.0,
    )

# Load FLAME config and model
# config = get_config()
config = get_default_config()
flame_model = FLAME(config).cuda()
faces = flame_model.faces

# Define input data model
class FlameParams(BaseModel):
    shape_params: List[float]
    expression_params: List[float]
    pose_params: List[float]

class InterpolationRequest(BaseModel):
    old_params: FlameParams
    new_params: FlameParams
    num_interpolations: int

@app.post("/generate_mesh/")
def generate_mesh(params: FlameParams):
    # Validate input data
    if len(params.shape_params) != config.shape_params:
        raise HTTPException(status_code=400, detail=f"Shape parameters must have {config.shape_params} values.")
    if len(params.expression_params) != config.expression_params:
        raise HTTPException(status_code=400, detail=f"Expression parameters must have {config.expression_params} values.")
    if len(params.pose_params) != config.pose_params:
        raise HTTPException(status_code=400, detail=f"Pose parameters must have {config.pose_params} values.")

    # Convert input data to tensors
    shape_params = torch.tensor([params.shape_params], dtype=torch.float32).repeat(config.batch_size, 1).cuda()
    expression_params = torch.tensor([params.expression_params], dtype=torch.float32).repeat(config.batch_size, 1).cuda()
    pose_params = torch.tensor([params.pose_params], dtype=torch.float32).repeat(config.batch_size, 1).cuda()

    # Forward pass to get vertices
    vertices, _ = flame_model(shape_params, expression_params, pose_params)
    vertices_np = vertices[0].detach().cpu().numpy()

    vertices_list = vertices_np.tolist()
    faces_list = faces.tolist()

    return JSONResponse(content={
        "vertices": vertices_list,
        "faces": faces_list
    })

@app.post("/generate_interpolated_meshes/")
def generate_interpolated_meshes(request: InterpolationRequest):
    # Parse input data
    old_params = request.old_params
    new_params = request.new_params
    num_interpolations = request.num_interpolations
    # Validate input data
    if len(old_params.shape_params) != config.shape_params or len(new_params.shape_params) != config.shape_params:
        raise HTTPException(status_code=400, detail=f"Shape parameters must have {config.shape_params} values.")
    if len(old_params.expression_params) != config.expression_params or len(new_params.expression_params) != config.expression_params:
        raise HTTPException(status_code=400, detail=f"Expression parameters must have {config.expression_params} values.")
    if len(old_params.pose_params) != config.pose_params or len(new_params.pose_params) != config.pose_params:
        raise HTTPException(status_code=400, detail=f"Pose parameters must have {config.pose_params} values.")
    if num_interpolations <= 0:
        raise HTTPException(status_code=400, detail="Number of interpolations must be greater than zero.")

    # Convert input data to numpy arrays
    old_shape = np.array(old_params.shape_params)
    new_shape = np.array(new_params.shape_params)
    old_expression = np.array(old_params.expression_params)
    new_expression = np.array(new_params.expression_params)
    old_pose = np.array(old_params.pose_params)
    new_pose = np.array(new_params.pose_params)

    # Generate interpolated parameters
    interpolated_meshes = []
    for i in range(1, num_interpolations + 1):
        t = i / (num_interpolations + 1)
        interp_shape = old_shape * (1 - t) + new_shape * t
        interp_expression = old_expression * (1 - t) + new_expression * t
        interp_pose = old_pose * (1 - t) + new_pose * t

        # Convert interpolated parameters to tensors
        shape_params = torch.tensor([interp_shape], dtype=torch.float32).repeat(config.batch_size, 1).cuda()
        expression_params = torch.tensor([interp_expression], dtype=torch.float32).repeat(config.batch_size, 1).cuda()
        pose_params = torch.tensor([interp_pose], dtype=torch.float32).repeat(config.batch_size, 1).cuda()

        # Forward pass to get vertices
        vertices, _ = flame_model(shape_params, expression_params, pose_params)
        vertices_np = vertices[0].detach().cpu().numpy()

        # Append mesh data
        interpolated_meshes.append({
            "vertices": vertices_np.tolist(),
            "faces": faces.tolist()
        })

    return JSONResponse(content=interpolated_meshes)


# Run server with uvicorn: uvicorn app:app --host 0.0.0.0 --port 8003