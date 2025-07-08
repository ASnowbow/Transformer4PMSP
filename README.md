# Transformer4PMSP

## Description

This repository contains the code for the paper "[A transformer-based deep reinforcement learning approach for
dynamic parallel machine scheduling problem with family setups](https://doi.org/10.1007/s10845-024-02470-8)":
The code is implemented in Python and uses PyTorch as the deep learning framework.

### Abstract

The parallel machine scheduling problem (PMSP) involves the optimized assignment of a set of jobs to a collection of
parallel machines, which is a proper formulation for the modern manufacturing environment. Deep reinforcement learning (
DRL) has been widely employed to solve PMSP. However, the majority of existing DRL-based frameworks still suffer from
generalizability and scalability. More specifically, the state and action design still heavily rely on human efforts. To
bridge these gaps, we propose a practical reinforcement learning-based framework to tackle a PMSP with new job arrivals
and family setup constraints. We design a variable-length state matrix containing full job and machine information. This
enables the DRL agent to autonomously extract features from raw data and make decisions with a global perspective. To
efficiently process this novel state matrix, we elaborately modify a Transformer model to represent the DRL agent. By
integrating the modified Transformer model to represent the DRL agent, a novel state representation can be effectively
leveraged. This innovative DRL framework offers a high-quality and robust solution that significantly reduces the
reliance on manual effort traditionally required in scheduling tasks. In the numerical experiment, the stability of the
proposed agent during training is first demonstrated. Then we compare this trained agent on 192 instances with several
existing approaches, namely a DRL-based approach, a metaheuristic algorithm, and a dispatching rule. The extensive
experimental results demonstrate the scalability of our approach and its effectiveness across a variety of scheduling
scenarios. Conclusively, our approach can thus solve the scheduling problems with high efficiency and flexibility,
paving the way for application of DRL in solving complex and dynamic scheduling problems.

## Installation

### Prerequisites

We recommend using [Anaconda](https://www.anaconda.com/) for all users for easier installation of Python packages and
required libraries. You need an environment with [Python3](https://www.python.org/) (>= 3.8).

### For Windows

Use the following commands in Command Prompt to install the required packages and clone the repository:

```bash
conda create -n pmsp python=3.8
conda activate pmsp
git clone https://github.com/ASnowbow/Transformer4PMSP.git
cd "*:\REPLACE\WITH\YOUR\PATH\TO\TRANSFORMER4PMSP"
# Use conda
conda env create -f pmsp.yml
# Or use pip
pip install -r requirements.txt
```

### For Linux and macOS

Use the following commands in Command Prompt to install the required packages and clone the repository:

```bash
conda create -n pmsp python=3.8
conda activate pmsp
git clone https://github.com/ASnowbow/Transformer4PMSP.git
cd Transformer4PMSP
# Use conda
conda env create -f pmsp.yml
# Or use pip
pip install -r requirements.txt
```

## How to use

### Training

You can simply train the model by running the following command in the terminal:

```bash
python main.py
```

This will start the training process using the default parameters defined in `config.py`. You can modify the parameters
in `config.py` to customize the training process, such as changing the number of episodes, batch size, learning rate,
and other hyperparameters.

### Evaluation

To evaluate the trained model, you can run the following command:

```bash
cd Validation
python env_val.py
```

This will load the trained model and evaluate it on the validation dataset. The results will directly be printed to the
console. You can modify the parameters in `config_val.py` to customize the evaluation process.

#### Model customization

To customize the model, you can modify the return of `experi_dir()` function in `config_val.py`. This function defines
the directory where the trained model is saved and loaded from. The format of the directory is as follows:

```
comparison\\{Number of}HeadAttention-{*}
```

Where:

- `{Number of}` is the number of heads in the multi-head attention mechanism.
- `{*}` is a placeholder for an indicator of global information usage.
    - The folder end with "1" is the agent trained wit the proposed state matrix.
    - The folder end with "0" is the agent trained with the comparative state matrix without global information.

#### Instance customization

To customize the instance, you can modify the `instance_folder` in `config_val.py`. This variable defines the folder in
`pmsp_instances` where the instances are stored. The format of the folder is as follows:

```
pmsp_instances\\r={*}_R={*}\\{*}m_{*}b_{*}new_{*}j_{*}f
```

Where:

- `{*}r` is the due date tightness
- `{*}R` is the due date range
- `{*}m` is the number of machines
- `{*}b` is the number of batches
- `{*}new` is the number of jobs per batch
- `{*}j` is the number of initial jobs
- `{*}f` is the number of families

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{li2024transformer,
  title={A transformer-based deep reinforcement learning approach for dynamic parallel machine scheduling problem with family setups},
  author={Li, Funing and Lang, Sebastian and Tian, Yuan and Hong, Bingyuan and Rolf, Benjamin and Noortwyck, Ruben and Schulz, Robert and Reggelin, Tobias},
  journal={Journal of Intelligent Manufacturing},
  pages={1--34},
  year={2024},
  publisher={Springer}
}
```

## Support

For questions regarding the code, or you want to contribute to this project, please open an issue or contact Li
via [email](mailto:funing.li@ift.uni-stuttgart.de).