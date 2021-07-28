# Installation

# Set up with Conda
```shell script
conda create -n py37 python=3.7 -y
conda activate py37
pip install torch==1.6.0 torchvision==0.7.0
pip install -r docs/requirements.txt
cd apex-master/
pip install -v --disable-pip-version-check --no-cache-dir ./
```
