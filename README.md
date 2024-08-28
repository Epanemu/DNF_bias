# Bias detection using DNF

Install requirements:

```shell
pip install -r requirements.txt
```

To use the provided code, one needs the Pyomo and Gurobi libraries (or modify the implementation to use a different solver)

To use the BRCG or Ripper through AIX360, install it separately:

```shell
conda create -n AIX360 python=3.10
conda activate AIX360
pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[rbm,rule_induction]
```

To use MDSS, install AIF360 library

```shell
pip install aif360
```
