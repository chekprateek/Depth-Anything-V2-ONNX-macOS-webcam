#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 16:15:00 2025

@author: prateeksrivastava
"""

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib

# Set up the color map
cmap = matplotlib.colormaps['Spectral_r']  # Updated colormap initialization

# Function to preprocess the input frame
def preprocess_frame(frame, input_size):
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    frame_transposed = np.transpose(frame_normalized, (2, 0, 1))
    return np.expand_dims(frame_transposed, axis=0)

# Function to postprocess the depth output
def postprocess_depth(depth_output, original_shape):
    depth = depth_output[0][0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)

    # Resize depth image to match original frame size
    depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))  # (width, height)
    return cmap(depth_resized)[:, :, :3] * 255

# Main function for webcam streaming and depth estimation
def main():
    input_size = 518
    model_path = 'checkpoints/depth_anything_model_vits.onnx'  # Path to your ONNX model

    session = ort.InferenceSession(model_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess_frame(raw_frame, input_size)
        ort_inputs = {session.get_inputs()[0].name: input_tensor}
        depth_output = session.run(None, ort_inputs)

        # Pass original shape for resizing later
        depth_image = postprocess_depth(depth_output, raw_frame.shape)

        combined_frame = np.hstack((raw_frame, depth_image.astype(np.uint8)))
        cv2.imshow('Webcam Depth Estimation', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

