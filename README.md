# LRMC (Learned Robust Matrix Completion)

## Overview
This repository implements the algorithm from the paper ["Deeply Learned Robust Matrix Completion for Large-scale Low-rank Data Recovery"](https://arxiv.org/pdf/2501.00677) by HanQin Cai, Chandra Kundu, Jialin Liu, and Wotao Yin.

## Abstract
Robust matrix completion (RMC) is a widely used machine learning tool that simultaneously tackles two critical issues in low-rank data analysis: missing data entries and extreme outliers. This paper proposes a novel scalable and learnable nonconvex approach, coined Learned Robust Matrix Completion (LRMC), for large-scale RMC problems. LRMC enjoys low computational complexity with linear convergence. Motivated by the proposed theorem, the free parameters of LRMC can be effectively learned via deep unfolding to achieve optimum performance. Furthermore, this paper proposes a flexible feedforward-recurrent-mixed neural network framework that extends deep unfolding from fixed-number iterations to infinite iterations. The superior empirical performance of LRMC is verified with extensive experiments against state-of-the-art methods on synthetic datasets and real applications, including video background subtraction, ultrasound imaging, face modeling, and cloud removal from satellite imagery.

## Repository Structure
- `lrmc_testing/`: Contains code for testing the LRMC algorithm on synthetic datasets
- `lrmc_training/`: Contains code for training the LRMC algorithm on synthetic datasets

## Getting Started
1. To test the LRMC algorithm:
    - Navigate to `lrmc_testing/`
    - Run `test_LRMC.m`

2. To train the LRMC algorithm:
    - Navigate to `lrmc_training/`
    - Install dependencies: `pip install -r requirements.txt`
    - Run `main.py`

## Citation
If you use this code, please cite:
```bibtex
@article{cai2024deeply,
     title={Deeply Learned Robust Matrix Completion for Large-scale Low-rank Data Recovery},
     author={Cai, HanQin and Kundu, Chandra and Liu, Jialin and Yin, Wotao},
     journal={arXiv preprint arXiv:2501.00677},
     year={2024}
}
```

## License
MIT
