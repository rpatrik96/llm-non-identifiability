<div align="center">    
 
# Understanding LLMs Requires More Than Statistical Generalization

[![Paper](http://img.shields.io/badge/arxiv-stat.ML:2405.01964-B31B1B.svg)](https://arxiv.org/abs/2405.01964)

[//]: # ([![Conference]&#40;http://img.shields.io/badge/CI4TS@UAI-2023.svg&#41;]&#40;https://sites.google.com/view/ci4ts2023/accepted-papers?authuser=0&#41;)

![CI testing](https://github.com/rpatrik96/llm-non-identifiability/workflows/CI%20testing/badge.svg?branch=main&event=push)

</div>
 
## Description   
Codebase for the ICML 2024 position paper titled _Understanding LLMs Requires More Than Statistical Generalization_.

## How to run

### Installing dependencies

```bash
# clone the repo with submodules  
git clone --recurse-submodules https://github.com/rpatrik96/llm-non-identifiability


# install the package   
cd llm-non-identifiability
pip install -e .   
pip install -r requirements.txt



# install requirements for tests
pip install --requirement tests/requirements.txt --quiet

# install pre-commit hooks (only necessary for development)
pre-commit install
 ```   

### Running on a SLURM cluster

```bash
# add execution permission to the scripts
chmod +x scripts/*.sh

# change userName in run_singularity_server.sh
userName=yourUserName

# specify root directory in export_root_dir.sh
export ROOT_DIR=/path/to/root/dir
export PACKAGE_NAME=/path/to/package/dir

# create the container file
cd scripts
singularity build --fakeroot nv.sif nv.def
```

### Weights and Biases sweep

```bash
# login to weights and biases
wandb login

# create sweep [spits out <sweepId>]
wandb sweep sweeps/<configFile>.yaml

# run sweep
./scripts/sweeps <sweepId>
```

## Citation   

```

@misc{reizinger2024understanding,
      title={Understanding LLMs Requires More Than Statistical Generalization}, 
      author={Patrik Reizinger and Szilvia Ujváry and Anna Mészáros and Anna Kerekes and Wieland Brendel and Ferenc Huszár},
      year={2024},
      eprint={2405.01964},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}

```   
