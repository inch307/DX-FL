# Federated Learning with Feedback Alignment

This repository contains the anonymized implementation for the ICCV 2025 submission **"Federated Learning with Feedback Alignment."**

## Dependencies
Ensure you have the necessary dependencies installed before running the experiments:
```bash
pip install -r requirements.txt
```
This code is compatible with Python 3.11.7
You may need to install the version of PyTorch that is compatible with your version.

## Cmds for Experiments

### FedAvg
```bash
python main.py --seed 0
```
### FedAvg with FLFA
```bash
python main.py --post_fa --seed 0
```

### FedAvgM
```bash
python main.py --seed 0 --server_momentum 0.1
```
### FedAvgM with FLFA
```bash
python main.py --seed 0 --server_momentum 0.1 --post_fa
```

### FedDecorr
```bash
python main.py --seed 0 --feddecorr --feddecorr_coef 0.1
```
### FedDecorr with FLFA
```bash
python main.py --seed 0 --feddecorr --feddecorr_coef 0.1 --post_fa
```

### FedProx
```bash
python main.py --alg fedprox --seed 0 --mu 0.1
```
### FedProx with FLFA
```bash
python main.py --alg fedprox -seed 0 --mu 0.1 --post_fa
```

### MOON
```bash
python main.py --alg moon --seed 0 --mu 0.1
```
### MOON with FLFA
```bash
python main.py --alg moon -seed 0 --mu 0.1 --post_fa
```
