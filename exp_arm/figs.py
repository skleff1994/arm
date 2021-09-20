
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
path = os.path.join(resultspath, 'trained_models/dvp/Order_1/Horizon_200/eps_19.pth')

# Load net 
Net  = torch.load(path)
DDPS_DATA =[]
WS = False
N=10
EPS_P = 0.3
# Sample test points
samples   =   samples_uniform_IK(nb_samples=N, eps_p=EPS_P, eps_v=0.0)
# Ref for warm start
ddp_ref = ocp_utils.init_DDP(robot, config, x0, critic=None, callbacks=False, which_costs=config['WHICH_COSTS'], dt=dt, N_h=N_h)
# Solve for several samples 
for k,x in enumerate(samples):
    robot.framesForwardKinematics(x[:nq])
    robot.computeJointJacobians(x[:nq])
    ddp = ocp_utils.init_DDP(robot, config, x, critic=Net, 
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt=dt, N_h=N_h) 
    ug = pin_utils.get_u_grav(q0, robot)
    ddp_ref.problem.x0 = x
    ddp_ref.solve( [x0 for i in range(N_h+1)] , [ug  for i in range(N_h)], maxiter=config['maxiter'], isFeasible=False)
    # Warm start using the croco ref
    xs_init = [ddp_ref.xs[i] for i in range(N_h+1)]
    us_init = [ddp_ref.us[i]  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=False)
    # Solve for each sample and record
    ddp_data = plot_utils.extract_ddp_data(ddp)
    DDPS_DATA.append(ddp_data)

def animate(data):
    viewer = robot.viz.viewer
    gui = viewer.gui
    import time
    for k,d in enumerate(data):
        print("Sample "+str(k)+"/"+str(len(data)))
        q = np.array(d['xs'])[:,:nq]
        for i in range(N_h+1):
            robot.display(q[i])
            time.sleep(dt)
