
import numpy as np
import utils.path_utils, utils.ocp_utils, utils.plot_utils, utils.pin_utils
from pinocchio.robot_wrapper import RobotWrapper
import torch
import sys
np.set_printoptions(precision=4, linewidth=180)
import matplotlib.pyplot as plt
import os
from datagen import samples_uniform_IK

# Load robot and OCP config
urdf_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'config/robot_properties_kuka/urdf/iiwa.urdf')
mesh_path = os.path.join(os.path.abspath(__file__ + "/../../"), 'config/robot_properties_kuka')
robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
config = utils.path_utils.load_config_file('static_reaching_task_ocp2')
q0 = np.asarray(config['q0'])
v0 = np.asarray(config['dq0'])
x0 = np.concatenate([q0, v0])   
robot.framesForwardKinematics(q0)
robot.computeJointJacobians(q0)
nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
N_h = config['N_h']
dt = config['dt']
id_ee = robot.model.getFrameId('contact')


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
    ddp = utils.ocp_utils.init_DDP(robot, config, x0, critic=Net, 
                                   callbacks=logs, 
                                   which_costs=config['WHICH_COSTS'],
                                   dt=dt, N_h=N_h) 
    ug = utils.pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    # Plot
    if(PLOT):
        utils.plot_utils.plot_ddp_results([ddp], robot, SHOW=True)
    return ddp


def test_trained_multiple(critic_path, N=20, PLOT=False):
    """
    Solve N OCPs using the trained NN as a terminal cost
    from sampled test points x0
    """
    # Sample test points
    samples  =   samples_uniform_IK(nb_samples=N, eps_p=0.05, eps_v=0.01)
    # Solve for each sample and record
    DDPS    =   [test_trained_single(critic_path, x0=x, PLOT=False, logs=False) for x in samples]
    # Plot results
    if(PLOT):
        utils.plot_utils.plot_ddp_results(DDPS, robot, SHOW=True, sampling_plot=1)
    return DDPS


def check_bellman(horizon=200, iter_number=1, WARM_START=0, PLOT=True):
    """
    Check that recursive property still holds on trained model: 
         - solve using croco over [0,..,(k+1)T]
         - solve using croco over [0,..,T]  + Vk
     where k = iter_number > 0 is the iteration number of the trained NN to be checked, i.e.
     when iter_number=1, we use eps_0.pth (a.k.a "V_1")
     when iter_number=2, we use eps_1.pth (a.k.a "V_2")
     ... 
     Should be the same  
    """
    # Solve OCP over [0,...,(k+1)T] using Crocoddyl
    ddp1 = utils.ocp_utils.init_DDP(robot, config, x0, critic=None, 
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt = dt, N_h=(iter_number+1)*N_h) 
    ug = utils.pin_utils.get_u_grav(q0, robot)
    xs_init = [x0 for i in range((iter_number+1)*N_h+1)]
    us_init = [ug  for i in range((iter_number+1)*N_h)]
    # Solve
    ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITHOUT CRITIC : Croco([0,...,"+ str(iter_number+1)+"T])")
    print("   Cost     = ", ddp1.cost)
    print("   V(xT)    = ", ddp1.problem.runningDatas[N_h].cost)
    print("   V_x(xT)  = \n", ddp1.problem.runningDatas[N_h].Lx)
    print("   V_xx(xT) = \n", ddp1.problem.runningDatas[N_h].Lxx)
    print("\n")

    # Solve OCP over [0,...,T] using k^th trained NN estimate as terminal model
    resultspath = os.path.join(os.path.abspath(__file__ + "/../../"), "results")
    critic_path = os.path.join(resultspath, f"trained_models/dvp/Order_{1}/Horizon_{horizon}/")
    critic_name = os.path.join(critic_path, "eps_"+str(iter_number-1)+".pth")
    print("Selecting trained network : eps_"+str(iter_number-1)+".pth\n")
    Net = torch.load(critic_name)
    ddp2 = utils.ocp_utils.init_DDP(robot, config, x0, critic=Net,
                                    callbacks=False, 
                                    which_costs=config['WHICH_COSTS'],
                                    dt = dt, N_h=N_h) 
    if(bool(WARM_START)):
        # Warm start using the croco ref
        xs_init = [ddp1.xs[i] for i in range(N_h+1)]
        us_init = [ddp1.us[i]  for i in range(N_h)]
    else:
        ug = utils.pin_utils.get_u_grav(q0, robot)
        xs_init = [x0 for i in range(N_h+1)]
        us_init = [ug  for i in range(N_h)]
    ddp2.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITH CRITIC of ITER #"+str(iter_number)+" :  Croco([0,...,T])+V_"+str(iter_number))
    if(bool(WARM_START)):
        print("  ( warm-started from Croco([0,..,"+str(iter_number+1)+"T]) )")
    print("  Cost = ", ddp2.cost)
    print("  V(xT)    = ", ddp2.problem.terminalData.cost)
    print("  V_x(xT)  = ", ddp2.problem.terminalData.Lx)
    print("  V_xx(xT) = \n", ddp2.problem.terminalData.Lxx)
    print("\n")
    # Plot
    if(PLOT):   
        # State 
        x1 = np.array(ddp1.xs); x2 = np.array(ddp2.xs)
        u1 = np.array(ddp1.us); u2 = np.array(ddp2.us)
        q1 = x1[:,:nq]; v1 = x1[:,nv:]
        q2 = x2[:,:nq]; v2 = x2[:,nv:] 
        fig_x, ax_x = plt.subplots(nq, 2, sharex='col') 
        fig_u, ax_u = plt.subplots(nu, 1, sharex='col') 
        if(bool(WARM_START)):
            label='Croco(0..T) + V_'+str(iter_number)+' ( warm-started from Croco([0,..,'+str(iter_number+1)+'T]) )'
        else:
            label='Croco(0..T) + V_'+str(iter_number)
        for i in range(nq):
            ax_x[i,0].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), q1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax_x[i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), q2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
            ax_x[i,0].grid(True)
            ax_x[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
            ax_x[i,1].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), v1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax_x[i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), v2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
            ax_x[i,1].grid(True)
            ax_x[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
            ax_u[i].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h), u1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax_u[i].plot(np.linspace(0*dt, N_h*dt, N_h), u2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
            ax_u[i].grid(True)
            ax_u[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax_x[-1,0].set_xlabel('Time (s)', fontsize=16)
        ax_x[-1,1].set_xlabel('Time (s)', fontsize=16)
        fig_x.align_ylabels(ax_x[:, 0])
        fig_x.align_ylabels(ax_x[:, 1])
        ax_u[-1].set_xlabel('Time (s)', fontsize=16)
        fig_u.align_ylabels(ax_u[:])
        handles_x, labels_x = ax_x[i,0].get_legend_handles_labels()
        fig_x.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        fig_x.suptitle('State : joint positions and velocities', fontsize=18)
        handles_u, labels_u = ax_u[0].get_legend_handles_labels()
        fig_u.legend(handles_u, labels_u, loc='upper right', prop={'size': 16})
        fig_u.suptitle('Control : joint torques', fontsize=18)

        # EE trajs
        p_ee1 = utils.pin_utils.get_p(q1, robot, id_ee)
        p_ee2 = utils.pin_utils.get_p(q2, robot, id_ee)
        v_ee1 = utils.pin_utils.get_v(q1, v1, robot, id_ee)
        v_ee2 = utils.pin_utils.get_v(q2, v2, robot, id_ee)
        fig_p, ax_p = plt.subplots(3, 2, sharex='col') 
        if(bool(WARM_START)):
            label='Croco(0..T) + V_'+str(iter_number)+' ( warm-started from Croco([0,..,'+str(iter_number+1)+'T]) )'
        else:
            label='Croco(0..T) + V_'+str(iter_number)
        xyz = ['x','y','z']
        for i in range(3):
            ax_p[i,0].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), p_ee1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax_p[i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), p_ee2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
            ax_p[i,0].grid(True)
            ax_p[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
            ax_p[i,1].plot(np.linspace(0*dt, (iter_number+1)*N_h*dt, (iter_number+1)*N_h+1), v_ee1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax_p[i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), v_ee2[:,i], linestyle='-', marker='o', color='r', label=label, alpha=0.5)
            ax_p[i,1].grid(True)
            ax_p[i,1].set_ylabel('$V^{EE}_%s$ (m/s)'%xyz[i], fontsize=16)
        ax_p[-1,0].set_xlabel('Time (s)', fontsize=16)
        ax_p[-1,1].set_xlabel('Time (s)', fontsize=16)
        fig_p.align_ylabels(ax_p[:,0])
        fig_p.align_ylabels(ax_p[:,1])
        handles_p, labels_p = ax_p[i,0].get_legend_handles_labels()
        fig_p.legend(handles_p, labels_p, loc='upper right', prop={'size': 16})
        fig_p.suptitle('End-effector positions and velocities', fontsize=18)
        plt.show()


if __name__=='__main__':
    # test_trained_single(sys.argv[1], int(sys.argv[2]))
    # test_trained_multiple(sys.argv[1], int(sys.argv[2]), int(sys.argv[-1]))
    check_bellman(sys.argv[1], int(sys.argv[2]), int(sys.argv[-1])) #, sys.argv[4])
