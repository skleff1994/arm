# What is it ?
Code for Differential Value Programming (DVP) implementation and example on the KUKA LBR IIWA 14 manipulator. 

# Dependencies
- [PyTorch](https://pytorch.org/) >= v1.8
- [Crocoddyl](https://github.com/loco-3d/crocoddyl) >= v1.8.1
- Python 3.6

# How to use it ?

Install PyTorch and Crocoddyl first if you don't have them already installed. The `git clone` this repo.

First generate a test set by running `datagen.py`, then launch the training using `main.py`. 


# Acknowledgements
You can find more information on this algorithm in this paper :

Parag, A., Kleff, S., Saci, L., Mansard, N., & Stasse, O. Value learning from trajectory optimization and Sobolev descent : A step toward reinforcement learning with superlinear convergence properties, _International Conference on Robotics and Automation (ICRA) 2022_ [submitted] 


