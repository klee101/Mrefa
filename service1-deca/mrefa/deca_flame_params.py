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

import os, sys
import argparse
import torch
from tqdm import tqdm
from scipy.io import savemat
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # Load image
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # Configure DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    # Process each image
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(images)  # Get FLAME parameters
            
        # Extract and save shape, expression, pose
        shape = codedict['shape'].cpu().numpy()
        expression = codedict['exp'].cpu().numpy()
        pose = codedict['pose'].cpu().numpy()

        # Save results as JSON
        result = {
            'shape': shape.tolist(),
            'expression': expression.tolist(),
            'pose': pose.tolist(),
        }
        if args.useTex:
            texture = codedict['tex'].cpu().numpy()
            result['texture'] = texture.tolist()
        
        with open(os.path.join(savefolder, f"{name}_params.json"), 'w') as f:
            json.dump(result, f, indent=4)

    print(f'-- FLAME parameters saved in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract FLAME parameters using DECA')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='Path to the test data, can be image folder, image path, image list, or video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='Path to the output directory, where results will be stored')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Set device, "cpu" for using CPU')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to crop input images. Set false if images are already well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='Detector for cropping face, check decalib/detectors.py for details')
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='Whether to extract FLAME texture parameters')

    main(parser.parse_args())