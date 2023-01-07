# AlphaRubiksCube

Simple AlphaZero-like approach to solving the Rubiks Cube

### Setup

```
conda install conda-build
conda env create -f environment.yml
conda activate alpha_rubiks_cube
conda develop .
```

Optional install cuda:

```
conda install -c nvidia -c pytorch cuda=11.7.1 pytorch-cuda=11.7
```
