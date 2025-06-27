# Based on DECA: Detailed Expression Capture and Animation (SIGGRAPH2021)
#  -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de
#
#
# Purpose of Use:
# This program is utilized for a course project titled "Mixed Reality Embodied Facial Animation" 
# by foxions from SJTU (Shanghai Jiao Tong University).

import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import numpy as np
from PIL import Image
import tempfile

from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

app = FastAPI()

# Initialize DECA model
device = "cuda"
deca_cfg.model.use_tex = True
deca_cfg.rasterizer_type = "standard"
deca = DECA(config=deca_cfg, device=device)

# Response model for FLAME parameters
class FlameParams(BaseModel):
    shape_params: list
    expression_params: list
    pose_params: list
    tex_params: list

@app.post("/deca-flame-params/", response_model=FlameParams)
async def deca_flame_params(file: UploadFile = File(...)):
    """
    API endpoint to process an image and return FLAME parameters.

    Args:
        file (UploadFile): Uploaded image file.

    Returns:
        FlameParams: JSON containing shape, expression, and pose parameters.
    """
    # Load and preprocess image
    input_image = Image.open(file.file).convert("RGB")
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        input_image.save(tmp.name)
        testdata = datasets.TestData(tmp.name, iscrop=True, face_detector='fan')
        data = testdata[0]
        input_tensor = data['image'].to(device)[None, ...]

        # Process image with DECA
        with torch.no_grad():
            codedict = deca.encode(input_tensor)

    # Extract parameters
    shape = np.squeeze(codedict['shape'].cpu().numpy()).tolist()
    expression = np.squeeze(codedict['exp'].cpu().numpy()).tolist()
    pose = np.squeeze(codedict['pose'].cpu().numpy()).tolist()
    tex = np.squeeze(codedict['tex'].cpu().numpy()).tolist()

    # Return as JSON
    return {"shape_params": shape, "expression_params": expression, "pose_params": pose, "tex_params": tex}

# Run server with uvicorn: uvicorn app:app --host 0.0.0.0 --port 8002