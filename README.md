# Federated Learning with Feedback Alignment

This repository contains the anonymized implementation for the ICCV 2025 submission **"Federated Learning with Feedback Alignment."**

## Dependencies
Ensure you have the necessary dependencies installed before running the experiments:
```bash
pip install -r requirements.txt
```
This code is compatible with Python 3.11.7

## Reproducing the Experiments

You can reproduce the experiments described in the paper using the following commands varying the seed (0-3):

### FedAvg
```bash
python main.py --dataset cifar10 --alg fedavg --model resnet50 --device cuda --epochs 10 --beta 0.3 --scheduler cosine --eta_min 0.001 --seed 0
python main.py --dataset cifar10 --alg fedavg --model resnet50 --device cuda --sync_round 1 --epochs 10 --beta 0.3 --post_fa --linear_scale --scheduler cosine --eta_min 0.001 --seed 0
```

### FedAvgM
```bash
python main.py --dataset cifar10 --alg fedavg --model resnet50 --device cuda --epochs 10 --beta 0.3 --scheduler cosine --eta_min 0.001 --seed 0 --server_momentum 0.1
python main.py --dataset cifar10 --alg fedavg --model resnet50 --device cuda --sync_round 1 --epochs 10 --beta 0.3 --post_fa --linear_scale --scheduler cosine --eta_min 0.001 --seed 0 --server_momentum 0.1
```
