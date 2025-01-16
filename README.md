# README for Monocular Depth Implementation

## Overview
This repository contains an implementation of monocular depth estimation using a macOS webcam. The project leverages pre-trained models to infer depth from video streams captured by the webcam.

## Requirements
Ensure you have the necessary dependencies installed. You can find them in the `requirements.txt` file. Install them using:

```bash
pip install -r requirements.txt
```

## Model Weights
The weights for the monocular depth model can be downloaded from the following link: [Depth Anything V2 Models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main?tab=readme-ov-file). 

Once downloaded, place the model `.pth` file into the `checkpoints/` folder.

## Conversion to ONNX
To convert the `.pth` weights into ONNX format, use the `path_to_onnx.py` script provided in this repository. This step is essential for utilizing the model in further applications.

## Running the Webcam Depth Estimation
After converting the weights to ONNX, you can use the `webcam_depth.py` script to perform depth estimation on video streams from your webcam.

### Usage Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Download and Place Model Weights**:
   Download the required model weights as mentioned above and place them in the `checkpoints/` directory.

3. **Convert Weights to ONNX**:
   Run the conversion script:
   ```bash
   python path_to_onnx.py
   ```

4. **Run Webcam Depth Estimation**:
   Start the depth estimation process using:
   ```bash
   python webcam_depth.py
   ```

## Future Work
- **Reduce Lag**: Work is ongoing to minimize latency when streaming video and performing monocular depth estimation (referred to as inferencing).
- **C++ Integration**: Future plans include utilizing ONNX in C++ for enhanced performance and flexibility.

## License
This project is licensed under [insert license here].

---

Feel free to modify any sections as needed, especially regarding licensing or additional instructions specific to your repository.

Citations:
[1] https://github.com/DepthAnything/Depth-Anything-V2/tree/main?tab=readme-ov-file
