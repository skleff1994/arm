# This files contains useful functions to plot results of OCP
# Author : Sébastien Kleff
# Date : 09/20/2021

import matplotlib.pyplot as plt
import numpy as np
import utils.pin_utils

# Extract relevant data from DDP solver for plotting
def extract_ddp_data(ddp):
    '''
    Record relevant data from ddp solver in order to plot 
    '''
    # Store data
    ddp_data = {}
    # OCP params
    ddp_data['T'] = ddp.problem.T
    ddp_data['dt'] = ddp.problem.runningModels[0].dt
    ddp_data['nq'] = ddp.problem.runningModels[0].state.nq
    ddp_data['nv'] = ddp.problem.runningModels[0].state.nv
    ddp_data['nu'] = ddp.problem.runningModels[0].differential.actuation.nu
    ddp_data['nx'] = ddp.problem.runningModels[0].state.nx
    # Pin model
    ddp_data['pin_model'] = ddp.problem.runningModels[0].differential.pinocchio
    ddp_data['frame_id'] = ddp.problem.runningModels[0].differential.costs.costs['translation'].cost.residual.id
    # Solution trajectories
    ddp_data['xs'] = ddp.xs
    ddp_data['us'] = ddp.us
    return ddp_data

# Plot results from DDP solver 
def plot_ddp_results(DDPS_DATA, which_plots='all', SHOW=False, sampling_plot=1):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      DDPS_DATA    : DDP solver data or list of ddp solvers data
      robot       : pinocchio robot wrapper
      name_endeff : name of end-effector (in pin model) 
    '''
    if(type(DDPS_DATA) != list):
        DDPS_DATA = [DDPS_DATA]
    for k,d in enumerate(DDPS_DATA):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(DDPS_DATA[k], SHOW=False)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(DDPS_DATA[k], SHOW=False)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(DDPS_DATA[k], SHOW=False)

        # Overlay on top of first plot
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all'):
                    plot_ddp_state(DDPS_DATA[k], fig=fig_x, ax=ax_x, SHOW=False, marker=None)
                if('u' in which_plots or which_plots =='all'):
                    plot_ddp_control(DDPS_DATA[k], fig=fig_u, ax=ax_u, SHOW=False, marker=None)
                if('p' in which_plots or which_plots =='all'):
                    plot_ddp_endeff(DDPS_DATA[k], fig=fig_p, ax=ax_p, SHOW=False, marker=None)

    if(SHOW):
      plt.show()
    
    fig = {}
    fig['p'] = fig_p
    fig['x'] = fig_x
    fig['u'] = fig_u

    ax = {}
    ax['p'] = ax_p
    ax['x'] = ax_x
    ax['u'] = ax_u

    return fig, ax
 
def plot_ddp_state(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (state)
    '''
    # Parameters
    N = ddp_data['T']
    dt = ddp_data['dt'] 
    nq = ddp_data['nq'] 
    nv = ddp_data['nv']
    x = np.array(ddp_data['xs'])
    # Extract pos, vel trajs
    q = x[:,:nq]
    v = x[:,nv:]
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nq, 2, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='State'
    for i in range(nq):
        # Positions
        ax[i,0].plot(tspan, q[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel('$q_%s$'%i, fontsize=16)
        ax[i,0].grid(True)
        # Velocities
        ax[i,1].plot(tspan, v[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel('$v_%s$'%i, fontsize=16)
        ax[i,1].grid(True)
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:, 0])
    fig.align_ylabels(ax[:, 1])
    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('State : joint positions and velocities', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (control)
    '''
    N = ddp_data['T'] 
    dt = ddp_data['dt'] 
    nu = ddp_data['nu'] 
    u = np.array(ddp_data['us'])
    # Plots
    tspan = np.linspace(0, N*dt-dt, N)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(nu, 1, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='Control'    
    for i in range(nu):
        # Positions
        ax[i].plot(tspan, u[:,i], linestyle='-', marker=marker, label=label)
        ax[i].set_ylabel('$u_%s$'%i, fontsize=16)
        ax[i].grid(True)
    ax[-1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:])
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('Control trajectories', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp_data, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp_data results (endeff)
    '''
    # Parameters
    N = ddp_data['T'] 
    dt = ddp_data['dt'] 
    nq = ddp_data['nq'] 
    x = np.array(ddp_data['xs'])
    # Extract EE traj
    q = x[:,:nq]
    v = x[:,nq:]
    p_EE = utils.pin_utils.get_p(q, ddp_data['pin_model'], ddp_data['frame_id'])
    v_EE = utils.pin_utils.get_v(q, v, ddp_data['pin_model'], ddp_data['frame_id'])
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='End-effector'
    xyz = ['x','y','z']
    for i in range(3):
        # Positions
        ax[i,0].plot(tspan, p_EE[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,0].grid(True)
        #Velocities
        ax[i,1].plot(tspan, v_EE[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel('$V^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax[i,1].grid(True)
    ax[-1,0].set_xlabel('Time (s)', fontsize=16)
    ax[-1,1].set_xlabel('Time (s)', fontsize=16)
    fig.align_ylabels(ax[:,0])
    fig.align_ylabels(ax[:,1])
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.suptitle('End-effector positions and velocities', size=18)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_refs(fig, ax, config):
    '''
    Overlay references on top of existing plots
    '''

    dt = config['dt']; N_h = config['N_h']
    nq = len(config['q0']); nu = nq
    # Add EE refs
    xyz = ['x','y','z']
    for i in range(3):
        ax['p'][i,0].plot(np.linspace(0, N_h*dt, N_h+1), [np.asarray(config['p_des']) [i]]*(N_h+1), 'r-.', label='Desired')
        ax['p'][i,0].set_ylabel('$P^{EE}_%s$ (m)'%xyz[i], fontsize=16)
        ax['p'][i,1].plot(np.linspace(0, N_h*dt, N_h+1), [np.asarray(config['v_des']) [i]]*(N_h+1), 'r-.', label='Desired')
        ax['p'][i,1].set_ylabel('$V^{EE}_%s$ (m)'%xyz[i], fontsize=16)
    handles_x, labels_x = ax['p'][i,0].get_legend_handles_labels()
    fig['p'].legend(handles_x, labels_x, loc='upper right', prop={'size': 16})

    # Add state refs
    for i in range(nq):
        ax['x'][i,0].plot(np.linspace(0*dt, N_h*dt, N_h+1), [np.asarray(config['q0'])[i]]*(N_h+1), 'r-.', label='Desired')
        ax['x'][i,1].plot(np.linspace(0*dt, N_h*dt, N_h+1), [np.asarray(config['dq0'])[i]]*(N_h+1), 'r-.', label='Desired')

    plt.show()
    
    # # Add torque refs
    # q = np.array(ddp_data['xs'])[:,:nq]
    # ureg_ref = np.zeros((N_h, nu))
    # for i in range(N_h):
    #     ureg_ref[i,:] = utils.pin_utils.get_u_grav_(q[i,:], ddp_data['pin_model'])
    # for i in range(nu):
    #     ax['u'][i].plot(np.linspace(0*dt, N_h*dt, N_h), ureg_ref[:,i], 'r-.', label='Desired')
