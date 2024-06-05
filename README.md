# Addictive-Congestion-Status(ACS)

## Description
The provided code is from the paper titled "Path Congestion Status Identification for Network Performance Tomography Using Deep Spatial-Temporal Learning." Here is an English summary of the contents in each folder:


## File Tree
The structure of this repository is following:
```
├── improve_nt:
│   ├── Topology_zoo.py       
│   ├── Topology_zoo_Analog.py
│   ├── alg_clink_2007.py     
│   ├── alg_clink_2007_pcs.py 
│   ├── alg_netscope.py       
│   ├── alg_netscope_pcs.py   
│   ├── alg_range_sum.py      
│   ├── alg_range_sum_pcs.py  
│   ├── evaluate.py
│   └── multi_threads.py
├── lstm-aae
│   ├── model.py        
│   ├── raw.py
│   └── val.py
└── ns-simulator   
    ├── Chinanet.cc
    ├── analyze.py 
    └── script3.sh 
```
Here is an English summary of the contents in each folder:

### Improve Network Tomography
This folder contains the implementation of three different classical network tomography algorithms. Files ending with `pcs` utilize ACS for result correction.

- **Topology_zoo.py**: Implements Boolean Network Tomography algorithms.
- **Topology_zoo_Analog.py**: Similar to BNT  but for Analog Network Tomography.
- **alg_clink_2007.py**: Implements the Clink algorithm for network tomography.
- **alg_clink_2007_pcs.py**: An adjusted version of `alg_clink_2007.py` using ACS for corrections.
- **alg_netscope.py**: Implements the Netscope algorithm for network tomography.
- **alg_netscope_pcs.py**: An adjusted version of `alg_netscope.py` using ACS for corrections.
- **alg_range_sum.py**: Implements the Range Sum algorithm for network tomography.
- **alg_range_sum_pcs.py**: An adjusted version of `alg_range_sum.py` using ACS for corrections.
- **evaluate.py**: Contains evaluation functions for assessing the performance of the algorithms.
- **multi_threads.py**: Implements multi-threading.

### LSTM-AAE
This folder contains the deep learning model for determining ACS.

- **model.py**: Contains the architecture and implementation of the LSTM-AAE model.
- **raw.py**: Handles raw data processing and preparation for the model.
- **val.py**: Contains validation scripts for testing the performance of the model.

### NS3-simulator
This folder contains actual network simulation data and related scripts.

- **Chinanet.cc**: Source file for simulating the ChinaNet network.
- **analyze.py**: Script for analyzing the simulation data.
- **script3.sh**: Shell script for running the simulation and analysis.