
import numpy as np
from utils import path_utils, ocp_utils, plot_utils, pin_utils
import torch
import sys
np.set_printoptions(precision=4, linewidth=180)
import matplotlib.pyplot as plt
import os
from datagen import samples_uniform_IK, samples_uniform
from robot_properties_kuka.config import IiwaConfig
from utils import path_utils
from test_trained import test_trained_multiple

config = path_utils.load_config_file('static_reaching_task_ocp2')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
robot = IiwaConfig.buildRobotWrapper()
robot.initDisplay(loadModel=True)
robot.display(q0)
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
N_h = config['N_h']
dt = config['dt']
id_ee = robot.model.getFrameId('contact')
resultspath = path_utils.results_path()

def test_trained_single(critic_path, PLOT=False, x0=x0, logs=True):
    """
    Solve an OCP using the trained NN as a terminal cost
    """
    # Load trained NN 
    Net  = torch.load(critic_path)
    # Init and solve
    q0 = x0[:nq]
    robot.framesForwardKinematics(q0)
    robot.computeJointJacobians(q0)
    ddp = ocp_utils.init_DDP(robot, config, x0, critic=Net, 
                                   callbacks=logs, 
                                   which_costs=config['WHICH_COSTS'],
                                   dt=dt, N_h=N_h) 
    ug = pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    ddp_data = plot_utils.extract_ddp_data(ddp)
    # Plot
    if(PLOT):
        fig, ax = plot_utils.plot_ddp_results(ddp_data, SHOW=False)
        plot_utils.plot_refs(fig, ax, config)
    return ddp_data

def test_trained_multiple(critic_path, N=20, PLOT=False):
    """
    Solve N OCPs using the trained NN as a terminal cost
    from sampled test points x0
    """
    # Sample test points
    samples   =   samples_uniform(nb_samples=N)
    # Solve for each sample and record
    DDPS_DATA = [test_trained_single(critic_path, x0=x, PLOT=False, logs=False) for x in samples]
    # Plot results
    if(PLOT):
        fig, ax = plot_utils.plot_ddp_results(DDPS_DATA, SHOW=False, sampling_plot=1)
        plot_utils.plot_refs(fig, ax, config)
    return DDPS_DATA, samples



# Fig. 1 : showing long crocoddyl trajectory matching a short croco+VF trajectory

# Fig. 2 : showing trajectories output by croco+VF

#  Video 1 : sampling training set in Gepetto Viewer

#  Video 2 : running a bunch of trajs in Gepetto Viewer
path = '/home/skleff/misc_repos/arm/results/trained_models/dvp/Order_1/Horizon_200/eps_19.pth'
DDPS_DATA, _ = test_trained_multiple(path, N=10, PLOT=False)

viewer = robot.viz.viewer
gui = viewer.gui
import time
# gui.addSphere('world/p_des', .02, [1. ,0 ,0, 1.])  
# gui.addBox('world/p_bounds',   2*eps_p, 2*eps_p, 2*eps_p,  [1., 1., 1., 0.3]) # depth(x),length(y),height(z), color
# tf_des = pin.utils.se3ToXYZQUAT(M_des)
# gui.applyConfiguration('world/p_des', tf_des)
# gui.applyConfiguration('world/p_bounds', tf_des)
# Check samples
for k,d in enumerate(DDPS_DATA):
    print("Sample "+str(k)+"/"+str(len(DDPS_DATA)))
    q = np.array(d['xs'])[:,:nq]
    for i in range(N_h+1):
        robot.display(q[i])
    # Update model and display sample
    # robot.framesForwardKinematics(sample[:nq])
    # robot.computeJointJacobians(sample[:nq])
    # M_ = robot.data.oMf[id_endeff]
    # tf_ = pin.utils.se3ToXYZQUAT(M_)
    # gui.applyConfiguration('world/sample'+str(k), tf_)
    # gui.refresh()
        time.sleep(dt)
