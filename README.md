# A Self-Improving Photosensitizer Discovery System via Bayesian Search Algorithm and First-Principle Simulation

## Code 
Code used for photosensitizer (PS) molecular space generation, graph convolutional model training, active learning, and other analysis purposes are found in this repository.

## Trained Models 
The DA and DAD PS prediction models can be found [here in this Google drive](https://drive.google.com/drive/folders/1Ir9Y7wcO-kfL1Ae2Zif6Hyn_uZzWEB64?usp=sharing).

## Data 
Our labelled dataset of 14168 photosensitizer structures are found in `data/Photosensitizers.csv`. All molecules are in SMILES format, S1-T1 energy gap (ST Gap) and HOMO LUMO gap (HL Gap) are both in eV. Ground states were optimized with b3lyp functional and 6-31G(d) basis set. Excited-state characteristics were calculated with TD-DFT with the same level of theory using the optimized ground state geometries. 

## Code Authors
[Jiali Li](https://github.com/jiali1025), [Pengfei Cai](https://github.com/cpfpengfei)
