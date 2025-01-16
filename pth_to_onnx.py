#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:01:57 2025

@author: prateeksrivastava
"""

import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

def convert_model_to_onnx():
    # Define the encoder type and input size
    encoder_type = 'vits'  # Change this if needed
    input_size = 518  # Default input size

    # Model configuration based on selected encoder
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Path to the .pth file
    model_path = os.path.join('checkpoints', f'depth_anything_v2_{encoder_type}.pth')

    # Load the model
    depth_anything = DepthAnythingV2(**model_configs[encoder_type])
    depth_anything.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_anything.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor
    input_tensor = torch.randn(1, 3, input_size, input_size)  # Batch size of 1 and RGB channels

    # Define output ONNX file path
    onnx_output_path = os.path.join('checkpoints', f'depth_anything_model_{encoder_type}.onnx')

    # Export the model to ONNX format
    torch.onnx.export(
        depth_anything,
        input_tensor,
        onnx_output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],   # Name of the input layer
        output_names=['output'],  # Name of the output layer
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic axes for batch size
    )

    print(f"Model has been converted to ONNX format and saved to {onnx_output_path}.")

# Run the conversion function
convert_model_to_onnx()
