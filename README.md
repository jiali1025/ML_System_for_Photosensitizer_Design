# Self-Improving Photosensitizer Discovery System via Bayesian Search with First-Principle Simulations

## Code 
Code used for photosensitizer (PS) molecular space generation, graph convolutional model training, active learning, and other analysis purposes are found in this repository.

## Trained Models 
The DA and DAD PS prediction models can be found [here in this Google drive](https://drive.google.com/drive/folders/1Ir9Y7wcO-kfL1Ae2Zif6Hyn_uZzWEB64?usp=sharing).

## Data 
Our labelled dataset of 14164 photosensitizer structures for both DA and DAD are found in `data/Photosensitizers_DA.csv` and `data/Photosensitizers_DAD.csv` respectively. All molecules are in SMILES format, and S1-T1 energy gap (ST Gap), HOMO LUMO gap (HL Gap), S1 and T1 are all in eV. Ground states were optimized with b3lyp functional and 6-31G(d) basis set. Excited-state characteristics were calculated with TD-DFT with the same level of theory using the optimized ground state geometries. 

## Code Authors
[Jiali Li](https://github.com/jiali1025), [Pengfei Cai](https://github.com/cpfpengfei)
