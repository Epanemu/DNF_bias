conda create -n AIX360 python=3.10
conda activate AIX260
pip install -e git+https://github.com/Trusted-AI/AIX360.git#egg=aix360[rbm,rule_induction]
