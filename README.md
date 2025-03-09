# PU-Label-Shift-Master-Thesis

Classification of Positive Unlabeled data under Label Shift.

## Setup Instructions

Follow these steps to set up the environment and install all necessary dependencies.

### 1. Clone the Repository

Clone this repository and the required forks to your local machine.

```bash
git clone https://github.com/izabelatelejko/PU-Label-Shift-Master-Thesis.git
cd PU-Label-Shift-Master-Thesis
git clone https://github.com/izabelatelejko/nnPUss.git src/nnPU
git clone https://github.com/izabelatelejko/DRPU.git src/DRPU
```

### 2. Create a Conda Environment

Create a new Conda environment named puls with Python 3.10.

```bash
conda create -y --name=puls python=3.10
```

### 3. Activate the Environment

```bash
conda activate puls
```

### Install Project Dependencies

Now install the required dependencies using pip. The following commands will install the dependencies, including the code from the forks (DRPU, nnPU).

```bash
pip install --no-index file://$(pwd)/src/nnPU
pip install --no-index file://$(pwd)/src/DRPU
pip install -r requirements.txt
```

### 5.  Install CUDA and PyTorch with GPU Support

To use GPU acceleration, install the required CUDA toolkit and PyTorch with CUDA 11.8 support. This can be done using the following Conda command:

```bash
conda install cudatoolkit=11.8 pytorch-cuda=11.8 -c nvidia -c pytorch -c conda-forge
```
