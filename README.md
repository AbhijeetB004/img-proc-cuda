# img-proc-cuda

This project investigates the performance gains achieved by parallelizing standard image processing algorithms using NVIDIA CUDA. It provides a comprehensive benchmarking suite comparing optimized GPU kernels against sequential CPU implementations (OpenCV/C++).

## Implemented Algorithms

The suite includes five distinct algorithms:

*   **KNN Denoising (K-Nearest Neighbors)**
    *   *Characteristic*: Compute-bound.
    *   *Implementation*: Global memory kernel vs CPU sequential.
*   **Non-Local Means (NLM) Denoising**
    *   *Characteristic*: Highly compute-intensive ($O(N^2)$ complexity relative to window size). Ideal for massive parallelization.
*   **Gaussian Convolution**
    *   *Characteristic*: Memory bandwidth dependent.
    *   *Optimization*: Includes a **Shared Memory** implementation to demonstrate L1 cache utilization and reduced global memory latency.
*   **Canny Edge Detection**
    *   *Characteristic*: Compound pipeline.
    *   *Pipeline*: KNN Denoising $\to$ Sobel Gradients $\to$ Hysteresis Thresholding. Verified for robustness on noisy inputs.
*   **Pixelization**
    *   *Characteristic*: Memory-bound / Low Arithmetic Intensity.
    *   *Purpose*: Demonstrates the "PCIe Bottleneck" where data transfer time outweighs computation time.

## Project Structure

```
.
├── src/
│   ├── cpu/            # Sequential C++ implementations (OpenCV based)
│   └── gpu/            # Parallel CUDA C++ implementations
├── notebooks/          # Jupyter notebooks for performance analysis
├── images/             # Test input data
└── analysis_output/    # Benchmarking artifacts (plots, processed images)
```

## Prerequisites & Setup

This project relies on system-level compilation tools. Ensure the following are installed before proceeding.

### 1. System Dependencies (Linux/WSL)
```bash
# Install Build Tools (CMake, Make, G++)
sudo apt-get update
sudo apt-get install build-essential cmake

# Install OpenCV (Required for Image I/O)
sudo apt-get install libopencv-dev
```
*   **CUDA Toolkit**: Must be installed separately from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads). Ensure `nvcc` is in your `$PATH`.

### 2. Python Environment
```bash
pip install -r requirements.txt
```

## Usage

### 1. Compilation
The provided Jupyter Notebooks automatically handle compilation steps (calling `cmake` and `make`) to adjust kernel parameters like Block Size dynamically. You do not need to build the project manually.

### 2. Input Data
Place high-resolution test images in the `images/` directory (already provided):
*   `input.jpg`: Clean reference image.
*   `input_noisy.jpg`: Noisy variant for denoising verification.

### 3. Benchmarking
Launch the analysis suite via Jupyter:
```bash
jupyter notebook
```
Execute the notebooks in `notebooks/` to compile the specific kernels and generate performance graphs.

## Benchmarking Metrics

The analysis notebooks enable the following verifications:

1.  **Visual Correctness**: Side-by-side comparison of CPU and GPU output images.
2.  **Speedup Scaling**: Performance ratio ($\frac{T_{cpu}}{T_{gpu}}$) mapped against increasing image resolutions ($512^2$ to $4096^2$).
3.  **Block Size Optimization**: Analysis of optimal CUDA `TILE_WIDTH` (Occupancy tuning) for the specific hardware architecture.

