# Maze_navigation

python code for maze navigation using Actor-Critic algorithm

## Installation

To use the code in this repository, simply clone the repo from github. 

Using `conda` is highly recommended.

create a conda env by

```bash
conda create -n your_env_name python==3.6
```

In your conda env, install the python dependencies for this repository.

```bash
source activate your_env_name
conda install numpy==1.16.4
conda install tensorflow==1.14.0
```

## Run the code
Run the code using terminal, such as:
```
python3 AC_maze_yu
```
Or by using your favorite IDE, such as pycharm


## Other scripts in this repo
Description of scripts:
Single maze traininig, export reward after training.
```
AC_maze_yu.py
```

Multi-maze traininig, export reward after training.
```
AC_maze_inference.py
```

The agent are able to choose between large acions and small actions.
```
AC_fast_action_maze_yu.py
```

Export value map after training
```
AC_valuemap_maze_yu.py
```

## Reference
The backbone of the Actor-Critic algorithm is based on code by Morvan Zhou.

https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/8_Actor_Critic_Advantage
