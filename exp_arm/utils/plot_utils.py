import matplotlib.pyplot as plt
import numpy as np
import utils.pin_utils

# Plot from DDP solver 
def plot_ddp_results(ddp, robot, name_endeff='contact', which_plots='all', SHOW=False, sampling_plot=1):
    '''
    Plot ddp results from 1 or several DDP solvers
    X, U, EE trajs
    INPUT 
      ddp         : DDP solver or list of ddp solvers
      robot       : pinocchio robot wrapper
      name_endeff : name of end-effector (in pin model) 
    '''
    if(type(ddp) != list):
        ddp = [ddp]
    for k,d in enumerate(ddp):
        # Return figs and axes object in case need to overlay new plots
        if(k==0):
            if('x' in which_plots or which_plots =='all'):
                fig_x, ax_x = plot_ddp_state(ddp[k], SHOW=False)
            if('u' in which_plots or which_plots =='all'):
                fig_u, ax_u = plot_ddp_control(ddp[k], SHOW=False)
            if('p' in which_plots or which_plots =='all'):
                fig_p, ax_p = plot_ddp_endeff(ddp[k], robot, name_endeff, SHOW=False)

        # Overlay on top of first plot
        else:
            if(k%sampling_plot==0):
                if('x' in which_plots or which_plots =='all'):
                    plot_ddp_state(ddp[k], fig=fig_x, ax=ax_x, SHOW=False, marker=None)
                if('u' in which_plots or which_plots =='all'):
                    plot_ddp_control(ddp[k], fig=fig_u, ax=ax_u, SHOW=False, marker=None)
                if('p' in which_plots or which_plots =='all'):
                    plot_ddp_endeff(ddp[k], robot, name_endeff, fig=fig_p, ax=ax_p, SHOW=False, marker=None)

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
 
def plot_ddp_state(ddp, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (state)
    '''
    # Parameters
    if(type(ddp)==dict):
      N = ddp['T'] #ddp.problem.T
      dt = ddp['dt'] #ddp.problem.runningModels[0].dt
      nq = ddp['nq'] #ddp.problem.runningModels[0].state.nq 
      nv = ddp['nv'] #ddp.problem.runningModels[0].state.nv
      x = np.array(ddp['xs'])
    else:
      N = ddp.problem.T
      dt = ddp.problem.runningModels[0].dt
      nq = ddp.problem.runningModels[0].state.nq
      nv = ddp.problem.runningModels[0].state.nv
      x = np.array(ddp.xs)
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

    # Legend
    handles, labels = ax[i,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('State trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_control(ddp, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (control)
    '''
    # Parameters
    if(type(ddp)==dict):
      N = ddp['T'] #ddp.problem.T
      dt = ddp['dt'] #ddp.problem.runningModels[0].dt
      nu = ddp['nu'] #ddp.problem.runningModels[0].state.nq
      u = np.array(ddp['us']) #np.array(ddp.us)
    else:
      N = ddp.problem.T
      dt = ddp.problem.runningModels[0].dt
      nu = ddp.problem.runningModels[0].state.nq
      u = np.array(ddp.us) #np.array(ddp.us)
    # Extract pos, vel trajs
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
        # Set xlabel on bottom plot
        if(i == nu-1):
            ax[i].set_xlabel('t (s)', fontsize=16)
    # Legend
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('Control trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

def plot_ddp_endeff(ddp, robot, name_endeff, fig=None, ax=None, label=None, SHOW=True, marker=None):
    '''
    Plot ddp results (endeff)
    '''
    # Parameters
    if(type(ddp)==dict):
      N = ddp['T'] #ddp.problem.T
      dt = ddp['dt'] #ddp.problem.runningModels[0].dt
      nq = ddp['nq'] #ddp.problem.runningModels[0].state.nq
      x = np.array(ddp['xs'])
    else:
      N = ddp.problem.T
      dt = ddp.problem.runningModels[0].dt
      nq = ddp.problem.runningModels[0].state.nq
      x = np.array(ddp.xs)
    # Extract EE traj
    q = x[:,:nq]
    v = x[:,nq:]
    id_endeff = robot.model.getFrameId(name_endeff)
    p = utils.pin_utils.get_p(q, robot, id_endeff)
    v_EE = utils.pin_utils.get_v(q, v, robot, id_endeff)
    # Plots
    tspan = np.linspace(0, N*dt, N+1)
    if(ax is None or fig is None):
        fig, ax = plt.subplots(3, 2, sharex='col', figsize=(19.2,10.8))
    if(label is None):
        label='End-effector'
    ylabels = ['Px', 'Py', 'Pz']
    for i in range(3):
        # Positions
        ax[i,0].plot(tspan, p[:,i], linestyle='-', marker=marker, label=label)
        ax[i,0].set_ylabel(ylabel=ylabels[i], fontsize=16)
        ax[i,0].grid(True)
        #Velocities
        ax[i,1].plot(tspan, v_EE[:,i], linestyle='-', marker=marker, label=label)
        ax[i,1].set_ylabel(ylabel=ylabels[i], fontsize=16)
        ax[i,1].grid(True)
    handles, labels = ax[i,0].get_legend_handles_labels()
    ax[i,0].set_xlabel('t (s)', fontsize=16)
    fig.legend(handles, labels, loc='upper right', prop={'size': 16})
    fig.align_ylabels()
    fig.suptitle('End-effector trajectories', size=16)
    if(SHOW):
        plt.show()
    return fig, ax

