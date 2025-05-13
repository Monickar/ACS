# Addictive-Congestion-Status (ACS)

## Project Description
This repository contains the implementation code for the paper titled **"Path Congestion Status Identification for Network Performance Tomography Using Deep Spatial-Temporal Learning."** The project introduces ACS, an innovative approach that enhances traditional network tomography by incorporating deep learning techniques to more accurately identify path congestion status in network environments.

## Project Overview
Network tomography is a methodology to infer internal network characteristics using end-to-end measurements. This project extends classical network tomography algorithms by incorporating a deep learning-based approach (ACS) to improve the accuracy of congestion detection and status identification.

The work is divided into three main components:
1. **Classical Network Tomography Algorithms** - Implementation of existing algorithms with ACS enhancements
2. **Deep Learning Model (LSTM-AAE)** - The core of the ACS approach using LSTM Adversarial Autoencoders
3. **Network Simulation** - NS-3 based simulation environment for testing and validation

## Repository Structure
```
├── improve_nt:               # Network Tomography Algorithm Implementations
│   ├── Topology_zoo.py       # Base implementation for Boolean Network Tomography
│   ├── Topology_zoo_Analog.py # Implementation for Analog Network Tomography
│   ├── alg_clink_2007.py     # Implementation of the Clink algorithm (INFOCOM'07)
│   ├── alg_clink_2007_pcs.py # Clink algorithm with ACS enhancement
│   ├── alg_netscope.py       # NetScope algorithm implementation
│   ├── alg_netscope_pcs.py   # NetScope algorithm with ACS enhancement
│   ├── alg_range_sum.py      # Range Sum algorithm implementation
│   ├── alg_range_sum_pcs.py  # Range Sum algorithm with ACS enhancement
│   ├── evaluate.py           # Evaluation scripts for algorithm performance
│   └── multi_threads.py      # Multi-threaded implementation for parallel testing
├── lstm-aae                  # Deep Learning Model for ACS
│   ├── model.py              # LSTM-AAE model architecture and training logic
│   ├── raw.py                # Data processing utilities
│   └── val.py                # Model validation scripts
└── ns-simulator              # Network Simulation Environment
    ├── Chinanet.cc           # NS-3 simulation for ChinaNet network topology
    ├── analyze.py            # Analysis scripts for simulation results
    └── script3.sh            # Shell script for batch simulation runs
```

## Key Components

### Network Tomography Algorithms (improve_nt)
The `improve_nt` folder contains implementations of classical network tomography algorithms with their ACS-enhanced versions:

- **Base Classes**: `Topology_zoo.py` and `Topology_zoo_Analog.py` provide the foundation for Boolean and Analog Network Tomography.
- **Network Tomography Algorithms**:
  - **Clink Algorithm**: Based on the paper "The Boolean Solution to the Congested IP Link Location Problem" (INFOCOM'07)
  - **NetScope Algorithm**: A method for network-wide inference of path metrics
  - **Range Sum Algorithm**: A technique for analyzing path delay characteristics

The files ending with `_pcs.py` are the enhanced versions that utilize ACS for result correction, significantly improving accuracy.

### Deep Learning Model (lstm-aae)
The LSTM-based Adversarial Autoencoder (LSTM-AAE) implementation is the core of the ACS approach:

- **Model Architecture**: Defined in `model.py`, the LSTM-AAE combines LSTM networks with adversarial training to better model temporal network characteristics.
- **Data Processing**: The `raw.py` file handles data preparation and preprocessing.
- **Validation**: Model performance evaluation through `val.py`.

The model is designed to understand and predict temporal patterns in network congestion, enabling more accurate status identification.

### Network Simulation (ns-simulator)
To validate the approach, a realistic network simulation environment is created:

- **ChinaNet Simulation**: `Chinanet.cc` contains a simulation of the ChinaNet network using NS-3.
- **Data Analysis**: `analyze.py` processes simulation results and prepares them for the ACS model.
- **Batch Testing**: `script3.sh` provides automation for running multiple simulations with varying parameters.

## Installation and Requirements

### Prerequisites
- Python 3.6+
- NS-3 (Network Simulator 3)
- PyTorch 1.7+
- NetworkX
- NumPy, SciPy, Matplotlib
- Gurobi Optimizer (for optimization components)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Monickar/ACS.git
   cd Addictive-Congestion-Status
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up NS-3 (if not already installed):
   ```bash
   # Follow NS-3 installation instructions from the official documentation
   # Once installed, copy the ns-simulator files to your NS-3 workspace
   ```

## Usage Examples

### Running Network Tomography Algorithms
```python
from improve_nt.Topology_zoo import 网络基类
from improve_nt.alg_clink_2007 import alg_clink_2007
from improve_nt.alg_clink_2007_pcs import alg_clink_2007_pcs

# Create a network instance
net = 网络基类()
net.配置拓扑("topology_zoo/topology_zoo数据集/Chinanet.gml")
net.部署测量路径(源节点列表=[2, 3])

# Run both standard and ACS-enhanced versions
results_standard = alg_clink_2007(Y, A_rm, x_pc)
results_enhanced = alg_clink_2007_pcs(Y, A_rm, x_pc, PCS)
```

### Training the LSTM-AAE Model
```python
from lstm-aae.model import get_model_single

# Train a model for single-path congestion detection
get_model_single(
    model_name="model_1", 
    dataset_name="path/to/training_data.pkl",
    seq_len=3,
    ratio=0.8,
    num_classes=3
)
```

### Running Network Simulations
```bash
# Run a single simulation with specific parameters
./waf --run "Chinanet --myParam=0.5 --probe_v=3.0 --anomoly_link=2"

# Run a batch of simulations using the provided script
./script3.sh 1.0 5.0 3 Chinanet
```

## Evaluation
The performance of different algorithms can be evaluated using the `evaluate.py` script:

```bash
python improve_nt/evaluate.py Chinanet 0.5 1 0 True True
```

This will run a comprehensive evaluation comparing classical algorithms against their ACS-enhanced versions, reporting metrics such as detection rate, false positive rate, precision, and F1-score.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Citation
If you use this code in your research, please cite the original paper:
```
@article{DU_NT_P_ACS,
title = {Identification of path congestion status for network performance tomography using deep spatial-temporal learning},
journal = {Computer Communications},
pages = {108194},
year = {2025},
issn = {0140-3664},
doi = {https://doi.org/10.1016/j.comcom.2025.108194},
url = {https://www.sciencedirect.com/science/article/pii/S0140366425001513},
author = {Chengze Du and Zhiwei Yu and Xiangyu Wang},
keywords = {Network tomography, Path congestion status, End-to-end measurement, LSTM, Adversarial autoencoders}
}
```

## Contributing
Contributions to this project are welcome. Please feel free to submit a Pull Request.

