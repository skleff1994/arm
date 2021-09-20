
import numpy as np
import utils_amit
from pinocchio.robot_wrapper import RobotWrapper
import torch
import sys
np.set_printoptions(precision=4, linewidth=180)
import matplotlib.pyplot as plt


def test_trained(critic_path, PLOT=True):
    """
    Solve an OCP using the trained NN as a terminal cost
    """
    # Read config file
    config = utils_amit.load_config_file('static_reaching_task_ocp')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])   
    # Get pin wrapper
    urdf_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka/urdf/iiwa.urdf'
    mesh_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka' 
    robot   =   RobotWrapper.BuildFromURDF(urdf_path, mesh_path)

    # Setup OCP with trained NN as terminal model
    N_h = config['N_h']
    dt = config['dt']
    Net  = torch.load(critic_path)
    ddp = utils_amit.init_DDP(robot, config, x0, critic=None,#Net, 
                                                callbacks=True, 
                                                which_costs=['translation', 
                                                             'ctrlReg', 
                                                             'stateReg', 
                                                             'stateLim'],
                                                dt = None, N_h=N_h) 
    # Warm-start
    ug = utils_amit.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(N_h+1)]
    us_init = [ug  for i in range(N_h)]
    # Solve
    ddp.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    if(PLOT):
        utils_amit.plot_ddp_results([ddp], robot, SHOW=True)

from datagen import samples

def test_(critic_path, N=20):
    """
    Solve N OCPs using the trained NN as a terminal cost
    from sampled test points x0
    """
    # Read config file
    config = utils_amit.load_config_file('static_reaching_task_ocp')
    # Get pin wrapper
    urdf_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka/urdf/iiwa.urdf'
    mesh_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka' 
    robot   =   RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
    nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
    N_h = config['N_h']
    dt = config['dt']
    # Sample test points
    points  =   samples(nb_samples=N, eps_p=0.5, eps_v=0.1)
    # Get trained NN to be tested
    Net  = torch.load(critic_path)
    DDPS    =   []
    for x0 in points:
        q0 = x0[:nq]
        robot.framesForwardKinematics(q0)
        robot.computeJointJacobians(q0)
        ddp = utils_amit.init_DDP(robot,
                                  config,
                                  x0,
                                  critic=Net,
                                  callbacks=False, 
                                  which_costs=['translation', 
                                               'ctrlReg', 
                                               'stateReg', 
                                               'stateLim'],
                                  dt=dt,
                                  N_h=N_h) 
        ddp.problem.x0  =   x0   
        ug = utils_amit.get_u_grav(q0, robot)
        xs_init = [x0 for i in range(N_h+1)]
        us_init = [ug  for i in range(N_h)]
        # Solve
        ddp.solve(xs_init, us_init, maxiter=1000, isFeasible=False)
        DDPS.append(ddp)

    # Plot results
    utils_amit.plot_ddp_results(DDPS, robot, SHOW=True, sampling_plot=1)


def check_bellman(critic_path, iter_number=1, WARM_START=0, PLOT=True):
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

    # Read config file
    config = utils_amit.load_config_file('static_reaching_task_ocp2')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])
    print("Initial state : ", x0)   
    # Get pin wrapper
    urdf_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka/urdf/iiwa.urdf'
    mesh_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka' 
    robot   =   RobotWrapper.BuildFromURDF(urdf_path, mesh_path)
    nq=robot.model.nq; nv=robot.model.nv; nu=nq; nx=nq+nv
    id_ee = robot.model.getFrameId('contact')

    # Setup OCP over [0,...,kT] and solve using Croco
    N_h = config['N_h']
    dt = config['dt']
    ddp1 = utils_amit.init_DDP(robot, config, x0, critic=None, 
                                                  callbacks=False, 
                                                  which_costs=['translation', 
                                                               'ctrlReg', 
                                                               'stateReg', 
                                                               'stateLim'],
                                                  dt = dt, N_h=(iter_number+1)*N_h) 
    ug = utils_amit.get_u_grav(q0, robot)
    xs_init = [x0 for i in range((iter_number+1)*N_h+1)]
    us_init = [ug  for i in range((iter_number+1)*N_h)]
    # Solve
    ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITHOUT CRITIC : Croco([0,...,"+ str(iter_number+1)+"T])")
    print("   Cost     = ", ddp1.cost)
    print("   V(xT)    = ", ddp1.problem.runningDatas[N_h].cost)
    print("   V_x(xT)  = \n", ddp1.problem.runningDatas[N_h].Lx)
    print("   V_xx(xT) = \n", ddp1.problem.runningDatas[N_h].Lxx)
    print("\n \n")
    # print("WS = ", bool(WARM_START))
    # Solve OCP over [0,...,T] using k^th trained NN estimate as terminal model
    critic_name = critic_path+"/eps_"+str(iter_number-1)+".pth"
    print("Selecting trained network : eps_"+str(iter_number-1)+".pth\n")
    Net = torch.load(critic_name)
    ddp2 = utils_amit.init_DDP(robot, config, x0, critic=Net,
                                                  callbacks=False, 
                                                  which_costs=['translation', 
                                                               'ctrlReg', 
                                                               'stateReg',
                                                               'stateLim' ],
                                                  dt = None, N_h=N_h) 
    if(bool(WARM_START)):
        # Warm start using the croco ref
        xs_init = [ddp1.xs[i] for i in range(N_h+1)]
        us_init = [ddp1.us[i]  for i in range(N_h)]
    else:
        ug = utils_amit.get_u_grav(q0, robot)
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
    print("\n \n")
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
        p_ee1 = utils_amit.get_p(q1, robot, id_ee)
        p_ee2 = utils_amit.get_p(q2, robot, id_ee)
        v_ee1 = utils_amit.get_v(q1, v1, robot, id_ee)
        v_ee2 = utils_amit.get_v(q2, v2, robot, id_ee)
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


def match_croco_V1(critic_path, PLOT=True):
    """
    Warm start croco(0..T)+V1 with half of croco(0..2T) + display V.F. +grads at xT
    """

    # Read config file
    config = utils_amit.load_config_file('static_reaching_task_ocp2')
    q0 = np.asarray(config['q0'])
    v0 = np.asarray(config['dq0'])
    x0 = np.concatenate([q0, v0])  
    print("Initial state : ", x0) 
    print("\n")
    # Get pin wrapper
    urdf_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka/urdf/iiwa.urdf'
    mesh_path = '/home/skleff/misc_repos/arm/exp_arm/robot_properties_kuka' 
    robot   =   RobotWrapper.BuildFromURDF(urdf_path, mesh_path)

    # Setup OCP over [0,...,2T] and solve using Croco
    N_h = config['N_h']
    dt = config['dt']
    ddp1 = utils_amit.init_DDP(robot, config, x0, critic=None, 
                                                  callbacks=False, 
                                                  which_costs=['translation', 
                                                               'ctrlReg', 
                                                               'stateReg', 
                                                               'stateLim'],
                                                  dt = dt, N_h=2*N_h) 
    # Warm-start with ug, x0
    ug = utils_amit.get_u_grav(q0, robot)
    xs_init = [x0 for i in range(2*N_h+1)]
    us_init = [ug  for i in range(2*N_h)]
    # Solve
    ddp1.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITHOUT CRITIC : Croco([0,...,2T])")
    print("   Cost     = ", ddp1.cost)
    print("   V(xT)    = ", ddp1.problem.runningDatas[N_h].cost)
    print("   V_x(xT)  = \n", ddp1.problem.runningDatas[N_h].Lx)
    print("   V_xx(xT) = \n", ddp1.problem.runningDatas[N_h].Lxx)
    print("\n \n")
    # Solve OCP over [0,...,T] using 1st trained NN estimate as terminal model
    critic_name = critic_path+"/eps_0.pth"
    print("Selecting trained network : eps_0.pth")
    Net = torch.load(critic_name)
    ddp2 = utils_amit.init_DDP(robot, config, x0, critic=Net,
                                                  callbacks=False, 
                                                  which_costs=['translation', 
                                                               'ctrlReg', 
                                                               'stateReg',
                                                               'stateLim' ],
                                                  dt = None, N_h=N_h) 
    # Warm start using the croco ref
    xs_init = [ddp1.xs[i] for i in range(N_h+1)]
    us_init = [ddp1.us[i]  for i in range(N_h)]
    ddp2.solve(xs_init, us_init, maxiter=config['maxiter'], isFeasible=True)
    print("WITH CRITIC of ITER #1 :  Croco([0,...,T])+V_1 warm-started with Croco(0,..,2T)")
    print("  Cost = ", ddp2.cost, " | V(xT) = ", ddp2.problem.terminalData.cost)
    print("  V(xT)    = ", ddp2.problem.terminalData.cost)
    print("  V_x(xT)  = ", ddp2.problem.terminalData.Lx)
    print("  V_xx(xT) = \n", ddp2.problem.terminalData.Lxx)
    # Plot
    if(PLOT):
        nq=7
        id_ee = robot.model.getFrameId('contact')
        p1 = utils_amit.get_p(np.array(ddp1.xs)[:,:nq], robot, id_ee)
        p2 = utils_amit.get_p(np.array(ddp2.xs)[:,:nq], robot, id_ee)
        fig, ax = plt.subplots(3, 1, sharex='col') 
        for i in range(3):
            # Plot a posteriori integration to check IAM
            ax[i].plot(np.linspace(0*dt, 2*N_h*dt, 2*N_h+1), p1[:,i], linestyle='-', marker='o', color='b', label='Croco', alpha=0.5)
            ax[i].plot(np.linspace(0*dt, N_h*dt, N_h+1), p2[:,i], linestyle='-', marker='o', color='r', label='Croco + V1 (warm-started)', alpha=0.5)
            ax[i].grid(True)
        handles_x, labels_x = ax[i].get_legend_handles_labels()
        fig.legend(handles_x, labels_x, loc='upper right', prop={'size': 16})
        plt.show()


if __name__=='__main__':
    # test_(sys.argv[1], int(sys.argv[2]))
    test_trained(sys.argv[1])
    # check_bellman(sys.argv[1], int(sys.argv[2]), int(sys.argv[-1])) #, sys.argv[4])
    # match_croco_V1(sys.argv[1])